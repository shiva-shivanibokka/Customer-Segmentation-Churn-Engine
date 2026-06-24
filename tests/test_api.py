"""Tests for the FastAPI scoring endpoint."""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient


VALID_PAYLOAD = {
    "Tenure": 12.0,
    "CityTier": 1,
    "WarehouseToHome": 15.0,
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 3,
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 1.0,
    "OrderCount": 3.0,
    "DaySinceLastOrder": 5.0,
    "CashbackAmount": 150.0,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory",
    "MaritalStatus": "Single",
}


def _make_mock_models():
    """Build a minimal models dict that mimics real artifacts for testing."""
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 13))

    mock_kmeans = MagicMock()
    mock_kmeans.predict.return_value = np.array([0])

    mock_clf = MagicMock()
    mock_clf.predict_proba.return_value = np.array([[0.4, 0.6]])

    mock_segment_models = {
        "Champions": {
            "calibrated_clf": mock_clf,
            "metrics": {"segment": "Champions"},
            "feature_cols": [],
        }
    }

    return {
        "segment_models": mock_segment_models,
        "kmeans": mock_kmeans,
        "scaler": mock_scaler,
    }


@pytest.fixture
def client():
    with patch("api.serve.get_models", return_value=_make_mock_models()):
        from api.serve import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestReadinessEndpoint:
    def test_readiness_returns_ready_when_models_available(self, client):
        response = client.get("/readiness")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestScoreEndpoint:
    def test_score_returns_200(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_score_response_has_required_fields(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        data = response.json()
        assert "segment" in data
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_tier" in data
        assert "customer_type" in data

    def test_churn_probability_is_float_in_range(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        prob = response.json()["churn_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_is_valid(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        tier = response.json()["risk_tier"]
        assert tier in ["Low Risk", "Medium Risk", "High Risk"]

    def test_customer_type_is_valid(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        ctype = response.json()["customer_type"]
        assert ctype in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Tenure"}
        response = client.post("/score", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_satisfaction_score_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "SatisfactionScore": 10}
        response = client.post("/score", json=bad_payload)
        assert response.status_code == 422
