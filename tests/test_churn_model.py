"""Tests for churn model scoring and classification logic."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_model import classify_customer_type
from churn_model import score_customers


class TestClassifyCustomerType:
    """Unit tests for the four-quadrant uplift classification."""

    def test_persuadable(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.50) == "Persuadable"

    def test_sure_thing(self):
        assert classify_customer_type(uplift_score=0.10, churn_prob=0.10) == "Sure Thing"

    def test_lost_cause(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.50) == "Lost Cause"

    def test_sleeping_dog(self):
        assert classify_customer_type(uplift_score=-0.10, churn_prob=0.10) == "Sleeping Dog"

    def test_boundary_churn_threshold(self):
        # Just below churn threshold (0.30 uses >=, so 0.29 is not high churn)
        result = classify_customer_type(uplift_score=0.10, churn_prob=0.29)
        assert result == "Sure Thing"

    def test_boundary_uplift_threshold(self):
        # Exactly at uplift threshold (0.05) — should be "positive uplift"
        result = classify_customer_type(uplift_score=0.05, churn_prob=0.50)
        assert result == "Persuadable"

    def test_custom_thresholds(self):
        result = classify_customer_type(
            uplift_score=0.03, churn_prob=0.50,
            uplift_threshold=0.02, churn_threshold=0.40,
        )
        assert result == "Persuadable"


class TestScoreCustomers:
    """Tests for score_customers using a mock model dict."""

    def _make_mock_segment_models(self):
        """Return a minimal mock that mimics the structure returned by train_segment_model."""
        from unittest.mock import MagicMock
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.4, 0.6]])
        return {
            "Champions": {
                "calibrated_clf": mock_clf,
                "feature_cols": ["EngagementScore", "RecencySignal"],
            }
        }

    def test_churn_probability_column_added(self):
        models = self._make_mock_segment_models()
        # Build a minimal one-row df
        df = pd.DataFrame({
            "Segment": ["Champions"],
            "EngagementScore": [0.7],
            "RecencySignal": [0.3],
        })
        result = score_customers(df, models, ["EngagementScore", "RecencySignal"])
        assert "ChurnProbability" in result.columns

    def test_risk_tier_assigned(self):
        models = self._make_mock_segment_models()
        df = pd.DataFrame({
            "Segment": ["Champions"],
            "EngagementScore": [0.7],
            "RecencySignal": [0.3],
        })
        result = score_customers(df, models, ["EngagementScore", "RecencySignal"])
        assert "RiskTier" in result.columns
        assert result["RiskTier"].iloc[0] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_high_prob_maps_to_high_risk(self):
        from unittest.mock import MagicMock
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% churn
        models = {"At-Risk": {"calibrated_clf": mock_clf, "feature_cols": ["EngagementScore"]}}
        df = pd.DataFrame({"Segment": ["At-Risk"], "EngagementScore": [0.1]})
        result = score_customers(df, models, ["EngagementScore"])
        assert result["RiskTier"].iloc[0] == "High Risk"
