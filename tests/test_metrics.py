"""
Unit tests for metrics calculations.

Tests all metric implementations to ensure correctness.
"""

import pytest
from metrics import MetricsCalculator, MetricResult


class TestMetricsCalculator:
    """Test suite for metrics calculations."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = ["hallucination", "grounded", "hallucination"]
        y_pred = ["hallucination", "grounded", "hallucination"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.accuracy()
        
        assert result.value == 1.0
        assert result.meets_target is True

    def test_accuracy_random(self):
        """Test accuracy with random predictions."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "grounded", "grounded", "hallucination"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.accuracy()
        
        assert result.value == 0.5

    def test_f1_perfect(self):
        """Test F1 with perfect predictions."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "grounded", "hallucination", "grounded"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.f1()
        
        assert result.value == 1.0

    def test_f1_all_positive(self):
        """Test F1 when predicting all positive."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "hallucination", "hallucination", "hallucination"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.f1()
        
        # TP=2, FP=2, FN=0
        # Precision = 2/(2+2) = 0.5
        # Recall = 2/(2+0) = 1.0
        # F1 = 2*(0.5*1.0)/(0.5+1.0) = 1.0/1.5 = 0.667
        assert pytest.approx(result.value, 0.01) == 0.667

    def test_precision(self):
        """Test precision calculation."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "hallucination", "hallucination", "grounded"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.precision()
        
        # Predict hallucination: indices 0,1,2
        # True hallucinations: indices 0,2
        # TP=2, FP=1
        # Precision = 2/(2+1) = 0.667
        assert pytest.approx(result.value, 0.01) == 0.667

    def test_recall(self):
        """Test recall calculation."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "grounded", "grounded", "grounded"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.recall()
        
        # True hallucinations: indices 0,2
        # Predicted hallucinations: index 0
        # TP=1, FN=1
        # Recall = 1/(1+1) = 0.5
        assert result.value == 0.5

    def test_tnr(self):
        """Test TNR (specificity) calculation."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "grounded", "hallucination", "hallucination"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.tnr()
        
        # True grounded: indices 1,3
        # Predicted grounded: index 1
        # TN=1, FP=1
        # TNR = 1/(1+1) = 0.5
        assert result.value == 0.5

    def test_bias_positive(self):
        """Test bias with more false positives."""
        y_true = ["hallucination", "hallucination", "grounded", "grounded"]
        y_pred = ["hallucination", "grounded", "hallucination", "grounded"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.bias()
        
        # TP=1, TN=1, FP=1, FN=1
        # Bias = (1-1)/4 = 0
        assert result.value == 0.0

    def test_cohens_kappa(self):
        """Test Cohen's kappa calculation."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"]
        y_pred = ["hallucination", "grounded", "hallucination", "grounded"]
        
        calc = MetricsCalculator(y_true, y_pred)
        result = calc.cohens_kappa()
        
        # Perfect agreement
        assert result.value == 1.0

    def test_metric_result_structure(self):
        """Test MetricResult dataclass structure."""
        result = MetricResult(
            name="test",
            value=0.85,
            intent="test intent",
            interpretation="test interpretation",
            target="≥ 0.75",
            meets_target=True,
            category="classification"
        )
        
        assert result.name == "test"
        assert result.value == 0.85
        assert result.meets_target is True

    def test_production_ready_pass(self):
        """Test production readiness check - pass."""
        y_true = ["hallucination", "grounded", "hallucination", "grounded"] * 10
        y_pred = ["hallucination", "grounded", "hallucination", "grounded"] * 10
        
        calc = MetricsCalculator(y_true, y_pred)
        is_ready, details = calc.production_ready()
        
        assert is_ready is True
        assert all(details.values())

    def test_production_ready_fail_f1(self):
        """Test production readiness check - fail on F1."""
        # Construct to have low F1
        y_true = ["hallucination"] * 10 + ["grounded"] * 10
        y_pred = ["grounded"] * 10 + ["hallucination"] * 10
        
        calc = MetricsCalculator(y_true, y_pred)
        is_ready, details = calc.production_ready()
        
        assert is_ready is False
        assert details["f1_pass"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
