"""
Hallucination Detection Metrics Module

This module implements all metrics for evaluating hallucination detection systems.
Each metric has documented intent, interpretation guide, and target values.

Metrics are organized by category:
- Agreement: Accuracy, Cohen's Kappa
- Classification: Precision, Recall, F1, TNR (Specificity)
- Correlation: Spearman, Pearson, Kendall's Tau (when confidence scores provided)
- Error: MAE, Bias, RMSE (when scores provided)
- Consistency: Variance across runs, Self-consistency rate
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)


@dataclass
class MetricResult:
    """
    Result container for a single metric evaluation.
    
    Attributes:
        name: Metric name (e.g., "f1", "precision", "tnr")
        value: Numeric metric value (0.0-1.0 for most)
        intent: Why we measure this metric
        interpretation: What different values mean
        target: Target value or range for production
        meets_target: Whether current value meets target
        category: Metric category (agreement, classification, correlation, error, consistency)
    """
    name: str
    value: float
    intent: str
    interpretation: str
    target: str
    meets_target: bool
    category: str
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary, handling nested dicts."""
        return asdict(self)


class MetricsCalculator:
    """
    Calculate all metrics for hallucination detection evaluation.
    
    Metrics are computed from predictions and ground truth labels.
    """

    # Interpretation Guides
    INTERPRETATIONS = {
        "accuracy": {
            "0.90+": "Excellent classification",
            "0.75-0.90": "Good, production-ready",
            "0.60-0.75": "Acceptable but needs improvement",
            "<0.60": "Poor, needs redesign"
        },
        "f1": {
            "0.85+": "Excellent balance",
            "0.75-0.85": "Good, production-ready",
            "<0.75": "Needs improvement"
        },
        "precision": {
            "0.90+": "Very few false hallucination alarms",
            "0.75-0.90": "Acceptable false alarm rate",
            "<0.75": "Users will distrust warnings"
        },
        "recall": {
            "0.90+": "Catch almost all hallucinations",
            "0.75-0.90": "Acceptable miss rate",
            "<0.75": "Too many undetected hallucinations"
        },
        "tnr": {
            "0.90+": "Rarely incorrectly flag grounded claims",
            "0.75-0.90": "Acceptable false alarm rate",
            "0.65-0.75": "Marginal, some false alarms",
            "<0.65": "Too aggressive, blocks good content"
        },
        "kappa": {
            "0.81+": "Almost perfect consistency",
            "0.61-0.80": "Substantial consistency",
            "0.41-0.60": "Moderate consistency",
            "0.21-0.40": "Fair consistency",
            "<0.20": "Poor consistency"
        },
        "bias": {
            ">0.15": "Biased toward over-predicting hallucinations (FP bias)",
            "-0.15 to 0.15": "Balanced",
            "<-0.15": "Biased toward under-predicting hallucinations (FN bias)"
        }
    }

    def __init__(self, y_true: List[str], y_pred: List[str], 
                 y_conf: Optional[List[float]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            y_true: Ground truth labels ("hallucination" or "grounded")
            y_pred: Predicted labels ("hallucination" or "grounded")
            y_conf: Optional confidence scores [0.0-1.0]
        """
        self.y_true = np.array([1 if label == "hallucination" else 0 for label in y_true])
        self.y_pred = np.array([1 if label == "hallucination" else 0 for label in y_pred])
        self.y_conf = np.array(y_conf) if y_conf else None
        
        # Validate inputs
        assert len(self.y_true) == len(self.y_pred), "Length mismatch"
        if self.y_conf is not None:
            assert len(self.y_conf) == len(self.y_true), "Confidence length mismatch"

    def _interpret_value(self, metric_name: str, value: float) -> str:
        """Get interpretation string for a metric value."""
        if metric_name not in self.INTERPRETATIONS:
            return f"Value: {value:.3f}"
        
        interpretations = self.INTERPRETATIONS[metric_name]
        for threshold, description in interpretations.items():
            try:
                if threshold.startswith(">"):
                    if value > float(threshold[1:]):
                        return description
                elif threshold.startswith("<"):
                    if value < float(threshold[1:]):
                        return description
                elif "-" in threshold and " to " in threshold:
                    parts = threshold.split(" to ")
                    low = float(parts[0])
                    high = float(parts[1])
                    if low <= value <= high:
                        return description
                elif threshold.startswith("+"):
                    if value >= float(threshold[:-1]):
                        return description
            except (ValueError, IndexError):
                # Skip if threshold can't be parsed as float
                continue
        
        return f"Value: {value:.3f}"

    # ========== AGREEMENT METRICS ==========

    def accuracy(self) -> MetricResult:
        """
        Accuracy: (TP + TN) / (TP + TN + FP + FN)
        
        Intent: Overall correctness across all cases
        Target: >= 0.75
        """
        value = accuracy_score(self.y_true, self.y_pred)
        interpretation = self._interpret_value("accuracy", value)
        target = "≥ 0.75"
        meets_target = value >= 0.75
        
        return MetricResult(
            name="accuracy",
            value=value,
            intent="Overall correctness across all cases; basic health check",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="agreement"
        )

    def cohens_kappa(self, y_true_run1: Optional[List[int]] = None, 
                      y_pred_run1: Optional[List[int]] = None) -> MetricResult:
        """
        Cohen's Kappa: Measures agreement correcting for chance.
        
        Can be used in two ways:
        1. Compare current predictions to ground truth
        2. Compare predictions from two runs (pass run1 predictions)
        
        Intent: Agreement between runs, correcting for chance; measure reproducibility
        Target: >= 0.70 across 3+ runs
        """
        if y_true_run1 is not None:
            # Compare two runs
            kappa = cohen_kappa_score(y_true_run1, self.y_pred)
        else:
            # Compare predictions to ground truth
            kappa = cohen_kappa_score(self.y_true, self.y_pred)
        
        interpretation = self._interpret_value("kappa", kappa)
        target = "≥ 0.70 (across 3+ runs)"
        meets_target = kappa >= 0.70
        
        return MetricResult(
            name="cohens_kappa",
            value=kappa,
            intent="Measure consistency/reproducibility between runs, correcting for chance",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="agreement"
        )

    # ========== CLASSIFICATION METRICS ==========

    def _compute_confusion_matrix(self) -> Tuple[int, int, int, int]:
        """
        Compute confusion matrix elements.
        
        Returns:
            (TP, TN, FP, FN)
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1]).ravel()
        return tp, tn, fp, fn

    def precision(self) -> MetricResult:
        """
        Precision: TP / (TP + FP)
        
        Intent: When we predict "hallucination", how often are we correct?
        When to use: If false positives (incorrectly flagging grounded claims) are costly
        Target: >= 0.75
        """
        value = precision_score(self.y_true, self.y_pred, zero_division=0)
        interpretation = self._interpret_value("precision", value)
        target = "≥ 0.75"
        meets_target = value >= 0.75
        
        return MetricResult(
            name="precision",
            value=value,
            intent="Of predicted hallucinations, how many are actually hallucinated?",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="classification"
        )

    def recall(self) -> MetricResult:
        """
        Recall (Sensitivity): TP / (TP + FN)
        
        Intent: Of actual hallucinations, how many do we catch?
        When to use: If false negatives (missing hallucinations) are dangerous
        Target: >= 0.75
        """
        value = recall_score(self.y_true, self.y_pred, zero_division=0)
        interpretation = self._interpret_value("recall", value)
        target = "≥ 0.75"
        meets_target = value >= 0.75
        
        return MetricResult(
            name="recall",
            value=value,
            intent="Of actual hallucinations, how many do we catch?",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="classification"
        )

    def f1(self) -> MetricResult:
        """
        F1 Score: 2 * (Precision × Recall) / (Precision + Recall)
        
        Intent: Harmonic mean of precision and recall; balanced correctness
        When to use: When both FP and FN costs are similar (primary metric)
        Target: >= 0.75 (minimum threshold)
        """
        value = f1_score(self.y_true, self.y_pred, zero_division=0)
        interpretation = self._interpret_value("f1", value)
        target = "≥ 0.75 (minimum threshold)"
        meets_target = value >= 0.75
        
        return MetricResult(
            name="f1",
            value=value,
            intent="Harmonic mean of precision and recall; balanced correctness",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="classification"
        )

    def tnr(self) -> MetricResult:
        """
        TNR (Specificity): TN / (TN + FP)
        
        Intent: Of grounded claims, how many do we correctly accept?
        When to use: Ensure system doesn't over-flag grounded content
        Target: >= 0.65 (minimum threshold)
        """
        tp, tn, fp, fn = self._compute_confusion_matrix()
        if (tn + fp) == 0:
            value = 0.0
        else:
            value = tn / (tn + fp)
        
        interpretation = self._interpret_value("tnr", value)
        target = "≥ 0.65 (minimum threshold)"
        meets_target = value >= 0.65
        
        return MetricResult(
            name="tnr",
            value=value,
            intent="Of grounded claims, how many do we correctly accept as grounded?",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="classification"
        )

    # ========== CORRELATION METRICS ==========

    def spearman_correlation(self) -> Optional[MetricResult]:
        """
        Spearman Correlation: Ranked correlation between variables.

        Intent: Measure monotonic relationship when model outputs confidence scores
        When to use: When LLM outputs confidence; check if confidence correlates with correctness
        Target: >= 0.70

        Returns None if confidence scores not provided or input is constant.
        """
        if self.y_conf is None:
            return None

        # Check for constant input (would cause undefined correlation)
        if np.std(self.y_conf) == 0 or np.std(self.y_true) == 0:
            return MetricResult(
                name="spearman_correlation",
                value=float('nan'),
                intent="Monotonic relationship between confidence and correctness",
                interpretation="Cannot compute: input values are constant (no variance)",
                target="≥ 0.70",
                meets_target=False,
                category="correlation",
                metadata={"error": "constant_input"}
            )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, pval = spearmanr(self.y_conf, self.y_true)

        # Handle NaN results
        if np.isnan(corr):
            return MetricResult(
                name="spearman_correlation",
                value=float('nan'),
                intent="Monotonic relationship between confidence and correctness",
                interpretation="Cannot compute: correlation undefined",
                target="≥ 0.70",
                meets_target=False,
                category="correlation",
                metadata={"error": "undefined"}
            )

        value = abs(corr)  # Use absolute value for interpretation

        interpretation = f"Correlation: {corr:.3f}, p-value: {pval:.4f}"
        if corr >= 0.70:
            interpretation += " (Strong monotonic relationship)"
        elif corr >= 0.60:
            interpretation += " (Moderate relationship)"
        else:
            interpretation += " (Weak relationship)"

        target = "≥ 0.70"
        meets_target = abs(corr) >= 0.70

        return MetricResult(
            name="spearman_correlation",
            value=value,
            intent="Monotonic relationship between confidence and correctness",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="correlation",
            metadata={"p_value": pval, "raw_correlation": corr}
        )

    def pearson_correlation(self) -> Optional[MetricResult]:
        """
        Pearson Correlation: Linear relationship between variables.

        Intent: Linear relationship between variables
        When to use: When assuming linear relationships; less robust than Spearman
        Target: >= 0.70

        Returns None if confidence scores not provided or input is constant.
        """
        if self.y_conf is None:
            return None

        # Check for constant input (would cause undefined correlation)
        if np.std(self.y_conf) == 0 or np.std(self.y_true) == 0:
            return MetricResult(
                name="pearson_correlation",
                value=float('nan'),
                intent="Linear relationship between confidence and correctness",
                interpretation="Cannot compute: input values are constant (no variance)",
                target="≥ 0.70",
                meets_target=False,
                category="correlation",
                metadata={"error": "constant_input"}
            )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, pval = pearsonr(self.y_conf, self.y_true)

        # Handle NaN results
        if np.isnan(corr):
            return MetricResult(
                name="pearson_correlation",
                value=float('nan'),
                intent="Linear relationship between confidence and correctness",
                interpretation="Cannot compute: correlation undefined",
                target="≥ 0.70",
                meets_target=False,
                category="correlation",
                metadata={"error": "undefined"}
            )

        value = abs(corr)

        interpretation = f"Correlation: {corr:.3f}, p-value: {pval:.4f}"
        if corr >= 0.70:
            interpretation += " (Strong linear relationship)"
        elif corr >= 0.60:
            interpretation += " (Moderate relationship)"
        else:
            interpretation += " (Weak relationship)"

        target = "≥ 0.70"
        meets_target = abs(corr) >= 0.70

        return MetricResult(
            name="pearson_correlation",
            value=value,
            intent="Linear relationship between confidence and correctness",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="correlation",
            metadata={"p_value": pval, "raw_correlation": corr}
        )

    def kendalls_tau(self) -> Optional[MetricResult]:
        """
        Kendall's Tau: Rank correlation based on concordant/discordant pairs.

        Intent: Measure ordinal association between confidence and correctness
        When to use: More robust than Spearman for small samples or many ties;
                     preferred for ordinal data or when distribution is unknown
        Target: >= 0.60 (tau values tend to be lower than Spearman/Pearson)

        Returns None if confidence scores not provided or input is constant.

        Interpretation:
        - tau = 1.0: Perfect agreement (all pairs concordant)
        - tau = 0.0: No association
        - tau = -1.0: Perfect disagreement (all pairs discordant)
        """
        if self.y_conf is None:
            return None

        # Check for constant input (would cause undefined correlation)
        if np.std(self.y_conf) == 0 or np.std(self.y_true) == 0:
            return MetricResult(
                name="kendalls_tau",
                value=float('nan'),
                intent="Ordinal association between confidence and correctness; robust to ties and outliers",
                interpretation="Cannot compute: input values are constant (no variance)",
                target="≥ 0.60",
                meets_target=False,
                category="correlation",
                metadata={"error": "constant_input"}
            )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau, pval = kendalltau(self.y_conf, self.y_true)

        # Handle NaN results
        if np.isnan(tau):
            return MetricResult(
                name="kendalls_tau",
                value=float('nan'),
                intent="Ordinal association between confidence and correctness; robust to ties and outliers",
                interpretation="Cannot compute: correlation undefined",
                target="≥ 0.60",
                meets_target=False,
                category="correlation",
                metadata={"error": "undefined"}
            )

        value = abs(tau)

        interpretation = f"Tau: {tau:.3f}, p-value: {pval:.4f}"
        if abs(tau) >= 0.70:
            interpretation += " (Strong ordinal association)"
        elif abs(tau) >= 0.50:
            interpretation += " (Moderate ordinal association)"
        elif abs(tau) >= 0.30:
            interpretation += " (Weak ordinal association)"
        else:
            interpretation += " (Very weak or no association)"

        target = "≥ 0.60"
        meets_target = abs(tau) >= 0.60

        return MetricResult(
            name="kendalls_tau",
            value=value,
            intent="Ordinal association between confidence and correctness; robust to ties and outliers",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="correlation",
            metadata={"p_value": pval, "raw_tau": tau}
        )

    # ========== ERROR METRICS ==========

    def mean_absolute_error(self) -> Optional[MetricResult]:
        """
        MAE: Average magnitude of prediction error on confidence scores.
        
        Intent: Measure confidence calibration quality
        When to use: When model outputs continuous scores
        Target: < 0.15
        
        Returns None if confidence scores not provided.
        """
        if self.y_conf is None:
            return None
        
        value = np.mean(np.abs(self.y_conf - self.y_true))
        
        interpretation = ""
        if value < 0.10:
            interpretation = "Excellent confidence calibration"
        elif value < 0.20:
            interpretation = "Acceptable calibration"
        else:
            interpretation = "Model is poorly calibrated"
        
        target = "< 0.15"
        meets_target = value < 0.15
        
        return MetricResult(
            name="mae",
            value=value,
            intent="Average magnitude of prediction error; calibration quality",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="error"
        )

    def root_mean_squared_error(self) -> Optional[MetricResult]:
        """
        RMSE: Penalizes large errors more heavily than MAE.
        
        Intent: Sensitive to outliers and large errors
        When to use: When large errors are particularly problematic
        Target: < 0.20
        
        Returns None if confidence scores not provided.
        """
        if self.y_conf is None:
            return None
        
        value = np.sqrt(np.mean((self.y_conf - self.y_true) ** 2))
        
        interpretation = ""
        if value < 0.15:
            interpretation = "Excellent error distribution"
        elif value < 0.25:
            interpretation = "Acceptable error distribution"
        else:
            interpretation = "Errors are too large"
        
        target = "< 0.20"
        meets_target = value < 0.20
        
        return MetricResult(
            name="rmse",
            value=value,
            intent="Penalizes large errors more heavily than MAE",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="error"
        )

    def bias(self) -> MetricResult:
        """
        Bias: (FP - FN) / Total or (Precision - Recall) / 2
        
        Intent: Detect systematic over/under-prediction
        When to use: Identify systematic patterns in errors
        Target: |bias| <= 0.15 (balanced)
        """
        tp, tn, fp, fn = self._compute_confusion_matrix()
        total = tp + tn + fp + fn
        
        if total == 0:
            value = 0.0
        else:
            value = (fp - fn) / total
        
        interpretation = self._interpret_value("bias", value)
        target = "|bias| ≤ 0.15 (balanced)"
        meets_target = abs(value) <= 0.15
        
        return MetricResult(
            name="bias",
            value=value,
            intent="Systematic over/under-prediction: positive=FP bias, negative=FN bias",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="error",
            metadata={"fp": fp, "fn": fn}
        )

    # ========== CONSISTENCY METRICS ==========

    def variance_across_runs(self, metric_values: List[float]) -> MetricResult:
        """
        Variance across multiple runs.
        
        Intent: How much does performance fluctuate run-to-run?
        When to use: Measure system reproducibility
        Target: < 0.05
        
        Args:
            metric_values: Metric values from 3+ runs
        """
        variance = np.var(metric_values)
        
        interpretation = ""
        if variance < 0.01:
            interpretation = "Highly stable across runs"
        elif variance < 0.05:
            interpretation = "Good stability"
        else:
            interpretation = "Poor stability; don't trust individual runs"
        
        target = "< 0.05"
        meets_target = variance < 0.05
        
        return MetricResult(
            name="variance",
            value=variance,
            intent="How much does performance fluctuate run-to-run?",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="consistency",
            metadata={"runs": len(metric_values), "mean": np.mean(metric_values)}
        )

    def self_consistency_rate(self, predictions_multiple_runs: List[List[int]]) -> MetricResult:
        """
        Self-consistency rate: Fraction of cases where model makes same decision across runs.
        
        Intent: Per-case consistency; does model always make same call?
        When to use: Identify ambiguous test cases; flag cases where model is uncertain
        Target: >= 0.90
        
        Args:
            predictions_multiple_runs: List of prediction arrays from 3+ runs
        """
        n_cases = len(predictions_multiple_runs[0])
        n_runs = len(predictions_multiple_runs)
        
        consistent_cases = 0
        for case_idx in range(n_cases):
            case_predictions = [run[case_idx] for run in predictions_multiple_runs]
            if len(set(case_predictions)) == 1:  # All same
                consistent_cases += 1
        
        value = consistent_cases / n_cases if n_cases > 0 else 0.0
        
        interpretation = ""
        if value >= 0.95:
            interpretation = "Almost always consistent"
        elif value >= 0.80:
            interpretation = "Acceptable; some borderline cases"
        else:
            interpretation = "Poor; high per-case variance"
        
        target = "≥ 0.90"
        meets_target = value >= 0.90
        
        return MetricResult(
            name="self_consistency",
            value=value,
            intent="Per-case consistency: does model make same decision across runs?",
            interpretation=interpretation,
            target=target,
            meets_target=meets_target,
            category="consistency",
            metadata={"consistent_cases": consistent_cases, "total_cases": n_cases}
        )

    # ========== AGGREGATE METHODS ==========

    def all_metrics(self) -> Dict[str, MetricResult]:
        """
        Calculate all available metrics.
        
        Returns dictionary of MetricResult objects.
        """
        metrics = {
            "accuracy": self.accuracy(),
            "cohens_kappa": self.cohens_kappa(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "tnr": self.tnr(),
            "bias": self.bias(),
        }
        
        # Optional correlation metrics (if confidence provided)
        if self.y_conf is not None:
            spearman = self.spearman_correlation()
            pearson = self.pearson_correlation()
            kendall = self.kendalls_tau()
            mae = self.mean_absolute_error()
            rmse = self.root_mean_squared_error()

            if spearman:
                metrics["spearman"] = spearman
            if pearson:
                metrics["pearson"] = pearson
            if kendall:
                metrics["kendalls_tau"] = kendall
            if mae:
                metrics["mae"] = mae
            if rmse:
                metrics["rmse"] = rmse

        return metrics

    def primary_metrics(self) -> Dict[str, MetricResult]:
        """
        Get only primary metrics for production threshold check.
        
        Primary metrics: F1, TNR, Bias
        These determine "go/no-go" for production.
        """
        return {
            "f1": self.f1(),
            "tnr": self.tnr(),
            "bias": self.bias(),
        }

    def production_ready(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if system meets production readiness thresholds.
        
        Thresholds:
        - F1 >= 0.75
        - TNR >= 0.65
        - |bias| <= 0.15
        
        Returns:
            (is_ready, details_dict)
        """
        f1_result = self.f1()
        tnr_result = self.tnr()
        bias_result = self.bias()
        
        details = {
            "f1_pass": f1_result.meets_target,
            "tnr_pass": tnr_result.meets_target,
            "bias_pass": bias_result.meets_target,
        }
        
        is_ready = all(details.values())
        
        return is_ready, details
