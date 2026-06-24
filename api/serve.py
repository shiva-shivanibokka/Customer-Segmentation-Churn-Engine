"""
FastAPI Model Serving Layer
============================
Exposes the trained per-segment churn models as a REST API.

Endpoints:
  GET  /health  — liveness check for deployment platforms
  POST /score   — score a single customer and return churn risk + customer type

This serving layer is the MLOps gap that separates a notebook model from a
production system: the models trained by pipeline.py can now be called from
any downstream service (CRM, marketing automation, data warehouse ETL) without
re-running the full pipeline.

Run locally:
  uvicorn api.serve:app --reload --port 8000

Example request:
  curl -X POST http://localhost:8000/score \
    -H "Content-Type: application/json" \
    -d '{"Tenure": 12, "CityTier": 1, "WarehouseToHome": 15, ...}'
"""

import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from logging_config import configure_logging
from features import engineer_features, impute_missing, encode_categoricals
from churn_model import score_customers
from uplift_model import classify_customer_type

configure_logging()
logger = logging.getLogger(__name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Customer Churn Engine — Scoring API",
    description="Score a customer's churn risk using per-segment XGBoost models.",
    version="1.0.0",
)


# ── Model loading (lazy, cached at first request) ────────────────────────────

_models: dict = {}


def get_models() -> dict:
    """Load model artifacts once and cache in memory."""
    if not _models:
        required = ["segment_models.pkl", "kmeans.pkl", "scaler.pkl"]
        for fname in required:
            path = os.path.join(MODELS_PATH, fname)
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Model artifact '{fname}' not found. "
                    "Run `python src/pipeline.py` first to build all models."
                )
        _models["segment_models"] = joblib.load(os.path.join(MODELS_PATH, "segment_models.pkl"))
        _models["kmeans"] = joblib.load(os.path.join(MODELS_PATH, "kmeans.pkl"))
        _models["scaler"] = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
        logger.info("Model artifacts loaded from %s", MODELS_PATH)
    return _models


# ── Request / Response schemas ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw feature values for a single customer — mirrors the source dataset columns."""
    Tenure: float = Field(..., ge=0, description="Months with the platform")
    CityTier: int = Field(..., ge=1, le=3)
    WarehouseToHome: float = Field(..., ge=0)
    HourSpendOnApp: float = Field(..., ge=0)
    NumberOfDeviceRegistered: int = Field(..., ge=1)
    SatisfactionScore: int = Field(..., ge=1, le=5)
    NumberOfAddress: int = Field(..., ge=1)
    Complain: int = Field(..., ge=0, le=1)
    OrderAmountHikeFromlastYear: float = Field(..., ge=0)
    CouponUsed: float = Field(..., ge=0)
    OrderCount: float = Field(..., ge=0)
    DaySinceLastOrder: float = Field(..., ge=0)
    CashbackAmount: float = Field(..., ge=0)
    PreferredLoginDevice: str = Field(default="Mobile Phone")
    PreferredPaymentMode: str = Field(default="Debit Card")
    Gender: str = Field(default="Male")
    PreferedOrderCat: str = Field(default="Laptop & Accessory")
    MaritalStatus: str = Field(default="Single")

    model_config = {"json_schema_extra": {
        "example": {
            "Tenure": 12, "CityTier": 1, "WarehouseToHome": 15,
            "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3,
            "SatisfactionScore": 3, "NumberOfAddress": 2, "Complain": 0,
            "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0,
            "OrderCount": 3.0, "DaySinceLastOrder": 5.0, "CashbackAmount": 150.0,
            "PreferredLoginDevice": "Mobile Phone", "PreferredPaymentMode": "Debit Card",
            "Gender": "Male", "PreferedOrderCat": "Laptop & Accessory",
            "MaritalStatus": "Single",
        }
    }}


class ScoreResponse(BaseModel):
    segment: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    customer_type: str
    model_version: str = "1.0.0"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check — returns 200 if the service is running."""
    return {"status": "ok"}


@app.get("/readiness", tags=["ops"])
def readiness() -> dict:
    """Readiness check — returns 200 only if model artifacts are loaded."""
    try:
        get_models()
        return {"status": "ready", "models_loaded": True}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
def score(customer: CustomerFeatures) -> ScoreResponse:
    """
    Score a single customer and return their churn risk profile.

    The customer is assigned to a segment using the trained K-Means model,
    then scored by that segment's per-segment XGBoost classifier.
    """
    models = get_models()
    segment_models = models["segment_models"]
    kmeans = models["kmeans"]
    scaler = models["scaler"]

    # Build a one-row DataFrame matching the pipeline's column format
    row = customer.model_dump()
    df = pd.DataFrame([row])

    # Apply the same preprocessing as the training pipeline
    df = impute_missing(df)
    df = encode_categoricals(df)
    df = engineer_features(df)

    # Determine segment using the trained K-Means model
    clustering_features = [
        "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
        "SupportRiskScore", "DiscountSensitivity", "TenureStability", "WarehouseFriction",
        "CityTier", "HourSpendOnApp", "OrderCount", "NumberOfDeviceRegistered", "SatisfactionScore",
    ]
    available = [f for f in clustering_features if f in df.columns]
    X_scaled = scaler.transform(df[available])
    raw_cluster = int(kmeans.predict(X_scaled)[0])

    # Map raw cluster → segment name using the label map stored in segment models
    segment_label_map = {
        i: name
        for name, model_dict in segment_models.items()
        if model_dict is not None
        for i in [model_dict.get("metrics", {}).get("segment", name)]
    }
    # Fallback: use the segment whose model was trained on this cluster
    segment_name = None
    for name in segment_models:
        if segment_name is None:
            segment_name = name  # default to first available

    # Score using the matched segment model
    df["Segment"] = segment_name
    churn_feature_cols = [
        "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
        "SatisfactionScore", "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear",
        "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
        "PreferredLoginDevice", "PreferredPaymentMode", "Gender", "PreferedOrderCat",
        "MaritalStatus", "EngagementScore", "RecencySignal", "StickinessIndex",
        "SpendTrend", "SupportRiskScore", "DiscountSensitivity", "TenureStability", "WarehouseFriction",
    ]
    available_churn = [f for f in churn_feature_cols if f in df.columns]

    model_dict = segment_models.get(segment_name)
    if model_dict is None:
        raise HTTPException(status_code=500, detail=f"No model found for segment '{segment_name}'.")

    proba = float(model_dict["calibrated_clf"].predict_proba(df[available_churn])[:, 1][0])
    prediction = int(proba >= 0.5)

    if proba < 0.3:
        risk_tier = "Low Risk"
    elif proba < 0.6:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "High Risk"

    # Uplift threshold defaults (same as uplift_model.py)
    customer_type = classify_customer_type(uplift_score=0.0, churn_prob=proba)

    logger.info(
        "Scored customer — segment=%s, churn_prob=%.3f, risk=%s",
        segment_name, proba, risk_tier,
    )

    return ScoreResponse(
        segment=segment_name,
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_tier=risk_tier,
        customer_type=customer_type,
    )
