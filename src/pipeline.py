"""
Pipeline Orchestrator
======================
Runs the full 5-stage decision intelligence pipeline end-to-end
and caches all artifacts to disk so the Streamlit UI loads instantly.

Run this once to build all models. The Streamlit app reads from cached
.parquet and .pkl files — no retraining on app load.
"""

import os
import sys
import joblib
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from features import build_pipeline, get_feature_sets
from segmentation import run_segmentation
from churn_model import run_churn_pipeline
from uplift_model import run_uplift_pipeline

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")


def run_full_pipeline(force_retrain: bool = False) -> dict:
    """
    Run all 5 stages of the decision intelligence pipeline.

    If cached artifacts exist and force_retrain=False, loads from disk.
    Set force_retrain=True to rebuild everything from scratch.
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    uplift_path = os.path.join(PROCESSED_PATH, "uplift.parquet")
    seg_profiles_path = os.path.join(MODELS_PATH, "segment_profiles.pkl")
    seg_models_path = os.path.join(MODELS_PATH, "segment_models.pkl")
    stability_path = os.path.join(MODELS_PATH, "stability.pkl")

    segmented_path = os.path.join(PROCESSED_PATH, "segmented.parquet")

    if not force_retrain and os.path.exists(uplift_path):
        print("[pipeline] Loading cached artifacts...")
        df = pd.read_parquet(uplift_path)
        seg_profiles = joblib.load(seg_profiles_path)
        segment_models = joblib.load(seg_models_path)
        stability = joblib.load(stability_path)
        feature_sets = get_feature_sets()
        print("[pipeline] Loaded from cache.")
        return {
            "df": df,
            "seg_profiles": seg_profiles,
            "segment_models": segment_models,
            "stability": stability,
            "feature_sets": feature_sets,
        }

    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION & CHURN ENGINE — FULL PIPELINE")
    print("=" * 60)

    # Stage 1: Feature Engineering
    print("\n[Stage 1] Feature Engineering")
    feature_sets = get_feature_sets()
    df = build_pipeline(save=True)

    # Stage 2: Segmentation — skip if cached segmented.parquet exists
    if (
        not force_retrain
        and os.path.exists(segmented_path)
        and os.path.exists(stability_path)
    ):
        print("\n[Stage 2] Customer Segmentation — loading from cache")
        df = pd.read_parquet(segmented_path)
        stability = joblib.load(stability_path)
        seg_profiles = (
            df.groupby("Segment")[
                [
                    "EngagementScore",
                    "RecencySignal",
                    "StickinessIndex",
                    "SpendTrend",
                    "SupportRiskScore",
                    "Churn",
                ]
            ]
            .mean()
            .round(3)
        )
        seg_profiles["CustomerCount"] = df.groupby("Segment").size()
        seg_profiles["ChurnRate"] = df.groupby("Segment")["Churn"].mean().round(3)
    else:
        print("\n[Stage 2] Customer Segmentation")
        seg_results = run_segmentation(df, feature_sets["clustering"], n_clusters=5)
        df = seg_results["df"]
        stability = seg_results["stability"]
        seg_profiles = seg_results["profiles"]
        joblib.dump(seg_profiles, seg_profiles_path)
        joblib.dump(stability, stability_path)

    # Stage 3: Per-Segment Churn Models
    print("\n[Stage 3] Per-Segment Churn Prediction")
    churn_results = run_churn_pipeline(df, feature_sets["churn_model"])
    df = churn_results["df"]
    segment_models = churn_results["segment_models"]

    # Stage 4: Uplift Modeling
    print("\n[Stage 4] Uplift Modeling (Causal ML)")
    uplift_results = run_uplift_pipeline(df, feature_sets["uplift_model"])
    df = uplift_results["df"]

    # Save final enriched dataset
    df.to_parquet(uplift_path, index=False)
    print(f"\n[pipeline] Pipeline complete. Final dataset: {df.shape}")
    print(f"[pipeline] Columns: {df.columns.tolist()}")

    return {
        "df": df,
        "seg_profiles": seg_profiles,
        "segment_models": segment_models,
        "stability": stability,
        "feature_sets": feature_sets,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain all models")
    args = parser.parse_args()

    results = run_full_pipeline(force_retrain=args.force)
    df = results["df"]

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total customers: {len(df)}")
    print(f"\nSegment distribution:")
    print(df["Segment"].value_counts())
    print(f"\nCustomer type distribution (uplift):")
    print(df["CustomerType"].value_counts())
    print(f"\nRisk tier distribution:")
    print(df["RiskTier"].value_counts())
    print(
        f"\nPersuadables with positive ROI: "
        f"{((df['CustomerType'] == 'Persuadable') & (df['NetROI'] > 0)).sum()}"
    )

    print("\nCluster stability:")
    stab = results["stability"]
    print(f"  Mean ARI: {stab['mean_ari']:.3f} ± {stab['std_ari']:.3f}")
    print(f"  Grade: {stab['grade']}")
