"""
Regression Testing: Validate on held-out test set with strict thresholds.

Ensures model doesn't degrade on held-out cases never used during development.
Thresholds are conservative to ensure production quality.
"""

from typing import Dict, List, Tuple
from metrics import MetricsCalculator


class RegressionTester:
    """
    Regression testing on held-out test set.
    """

    # Production thresholds - conservative and strict
    PRODUCTION_THRESHOLDS = {
        "f1": 0.75,           # Minimum 0.75 F1
        "tnr": 0.65,          # Minimum 65% specificity
        "bias": 0.15,         # Maximum 15% bias magnitude
    }

    def run_regression_test(self, results: List[Dict]) -> Tuple[bool, Dict[str, any]]:
        """
        Run regression test on held-out set.
        
        Args:
            results: Evaluation results on regression test set
            
        Returns:
            (passed, details) where passed is True if all thresholds met
        """
        # Extract ground truth and predictions
        y_true = [r["ground_truth"] for r in results]
        y_pred = [r["prediction"] for r in results]
        
        # Calculate metrics
        calc = MetricsCalculator(y_true, y_pred)
        f1_result = calc.f1()
        tnr_result = calc.tnr()
        bias_result = calc.bias()
        
        # Check thresholds
        details = {
            "f1": {
                "value": f1_result.value,
                "threshold": self.PRODUCTION_THRESHOLDS["f1"],
                "pass": f1_result.value >= self.PRODUCTION_THRESHOLDS["f1"],
                "reason": f"F1 must be >= {self.PRODUCTION_THRESHOLDS['f1']}"
            },
            "tnr": {
                "value": tnr_result.value,
                "threshold": self.PRODUCTION_THRESHOLDS["tnr"],
                "pass": tnr_result.value >= self.PRODUCTION_THRESHOLDS["tnr"],
                "reason": f"TNR must be >= {self.PRODUCTION_THRESHOLDS['tnr']}"
            },
            "bias": {
                "value": abs(bias_result.value),
                "threshold": self.PRODUCTION_THRESHOLDS["bias"],
                "pass": abs(bias_result.value) <= self.PRODUCTION_THRESHOLDS["bias"],
                "reason": f"|Bias| must be <= {self.PRODUCTION_THRESHOLDS['bias']}"
            }
        }
        
        # Overall pass: all metrics pass
        passed = all(d["pass"] for d in details.values())
        
        return passed, details

    def format_report(self, passed: bool, details: Dict) -> str:
        """
        Format regression test report as readable string.
        
        Args:
            passed: Whether regression test passed
            details: Details from run_regression_test()
            
        Returns:
            Formatted report string
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        report = f"\n{'='*60}\nREGRESSION TEST REPORT: {status}\n{'='*60}\n\n"
        
        for metric_name, metric_details in details.items():
            status_char = "✓" if metric_details["pass"] else "✗"
            value = metric_details["value"]
            threshold = metric_details["threshold"]
            
            report += f"{status_char} {metric_name.upper()}: {value:.3f}"
            report += f" (threshold: {threshold})\n"
            report += f"  {metric_details['reason']}\n\n"
        
        report += "="*60 + "\n"
        
        if not passed:
            failed_metrics = [name for name, d in details.items() if not d["pass"]]
            report += f"\nFailed metrics: {', '.join(failed_metrics)}\n"
            report += "Action: Fix identified issues before production deployment\n"
        else:
            report += "\n✓ All thresholds met; ready for production\n"
        
        return report
