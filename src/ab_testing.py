"""
A/B Testing Framework: Compare prompt versions and determine winners.

Handles:
- Running both prompts on same test set
- Calculating metrics for each
- Statistical significance testing
- Recording results with winner determination
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from scipy import stats

from metrics import MetricsCalculator


class ABTester:
    """
    Framework for comparing two prompt versions.
    """

    def __init__(self, prompt_registry_path: str = "prompts/prompt_registry.json"):
        """
        Initialize A/B tester.
        
        Args:
            prompt_registry_path: Path to prompt registry JSON
        """
        self.prompt_registry_path = Path(prompt_registry_path)
        self._load_registry()

    def _load_registry(self) -> None:
        """Load prompt registry."""
        with open(self.prompt_registry_path, "r") as f:
            self.registry = json.load(f)

    def _save_registry(self) -> None:
        """Save updated registry."""
        with open(self.prompt_registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def run_test(self,
                 results_a: List[Dict],
                 results_b: List[Dict],
                 prompt_a_id: str,
                 prompt_b_id: str,
                 test_name: Optional[str] = None,
                 alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compare two sets of evaluation results.
        
        Args:
            results_a: Results from prompt A (list of dicts with 'ground_truth', 'prediction')
            results_b: Results from prompt B
            prompt_a_id: ID of prompt A (from registry)
            prompt_b_id: ID of prompt B
            test_name: Optional name for this test
            alpha: Significance level (default 0.05)
            
        Returns:
            Test results with winner, metrics, and significance
        """
        # Extract ground truth and predictions
        y_true_a = [r["ground_truth"] for r in results_a]
        y_pred_a = [r["prediction"] for r in results_a]
        
        y_true_b = [r["ground_truth"] for r in results_b]
        y_pred_b = [r["prediction"] for r in results_b]
        
        # Calculate metrics for both
        metrics_a = MetricsCalculator(y_true_a, y_pred_a).all_metrics()
        metrics_b = MetricsCalculator(y_true_b, y_pred_b).all_metrics()
        
        # Get primary metrics
        f1_a = metrics_a["f1"].value
        f1_b = metrics_b["f1"].value
        
        tnr_a = metrics_a["tnr"].value
        tnr_b = metrics_b["tnr"].value
        
        # McNemar's test for statistical significance
        mcnemar_stat, mcnemar_pval = self._mcnemars_test(results_a, results_b)
        
        # Determine winner
        winner = self._determine_winner(metrics_a, metrics_b)
        
        test_result = {
            "test_id": test_name or f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "prompt_a": prompt_a_id,
            "prompt_b": prompt_b_id,
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(results_a),
            "metrics_a": {k: v.to_dict() for k, v in metrics_a.items()},
            "metrics_b": {k: v.to_dict() for k, v in metrics_b.items()},
            "winner": winner,
            "significance": {
                "mcnemar_statistic": mcnemar_stat,
                "p_value": mcnemar_pval,
                "is_significant": mcnemar_pval < alpha
            },
            "summary": {
                "f1_a": f1_a,
                "f1_b": f1_b,
                "f1_improvement": f1_b - f1_a,
                "tnr_a": tnr_a,
                "tnr_b": tnr_b,
                "tnr_improvement": tnr_b - tnr_a
            }
        }
        
        return test_result

    def _mcnemars_test(self, results_a: List[Dict], 
                       results_b: List[Dict]) -> Tuple[float, float]:
        """
        McNemar's test for comparing paired classifiers.
        
        Returns:
            (chi2_statistic, p_value)
        """
        # Count cases where predictions differ
        disagreements = {"a_correct_b_wrong": 0, "a_wrong_b_correct": 0}
        
        for res_a, res_b in zip(results_a, results_b):
            a_correct = res_a["correct"]
            b_correct = res_b["correct"]
            
            if a_correct and not b_correct:
                disagreements["a_correct_b_wrong"] += 1
            elif not a_correct and b_correct:
                disagreements["a_wrong_b_correct"] += 1
        
        # McNemar's test
        n01 = disagreements["a_correct_b_wrong"]
        n10 = disagreements["a_wrong_b_correct"]
        
        if (n01 + n10) == 0:
            return 0.0, 1.0
        
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        pval = 1 - stats.chi2.cdf(chi2, df=1)
        
        return chi2, pval

    def _determine_winner(self, metrics_a: Dict, metrics_b: Dict) -> str:
        """
        Determine winner based on primary metrics.
        
        Primary metrics: F1, TNR, Bias
        Wins if: higher F1, higher TNR, lower |bias|
        """
        f1_a = metrics_a["f1"].value
        f1_b = metrics_b["f1"].value
        
        tnr_a = metrics_a["tnr"].value
        tnr_b = metrics_b["tnr"].value
        
        bias_a = abs(metrics_a["bias"].value)
        bias_b = abs(metrics_b["bias"].value)
        
        # Score each prompt (higher is better)
        score_a = 0
        score_b = 0
        
        if f1_a > f1_b:
            score_a += 1
        elif f1_b > f1_a:
            score_b += 1
        
        if tnr_a > tnr_b:
            score_a += 1
        elif tnr_b > tnr_a:
            score_b += 1
        
        if bias_a < bias_b:
            score_a += 1
        elif bias_b < bias_a:
            score_b += 1
        
        if score_a > score_b:
            return "a"
        elif score_b > score_a:
            return "b"
        else:
            return "tie"

    def update_registry_with_results(self, test_result: Dict) -> None:
        """
        Update prompt registry with test results.
        
        Args:
            test_result: Result from run_test()
        """
        # Store test in registry
        self.registry["ab_tests"].append({
            "test_id": test_result["test_id"],
            "prompt_a": test_result["prompt_a"],
            "prompt_b": test_result["prompt_b"],
            "status": "completed",
            "winner": test_result["winner"],
            "timestamp": test_result["timestamp"],
            "metrics": test_result["summary"]
        })
        
        # Update prompt results if data available
        if "metrics_a" in test_result:
            prompt_a_idx = next(
                (i for i, p in enumerate(self.registry["prompts"]) 
                 if p["id"] == test_result["prompt_a"]),
                None
            )
            if prompt_a_idx is not None and test_result["metrics_a"]:
                # Store F1, TNR, Bias
                self.registry["prompts"][prompt_a_idx]["results"] = {
                    "f1": test_result["metrics_a"].get("f1", {}).get("value"),
                    "tnr": test_result["metrics_a"].get("tnr", {}).get("value"),
                    "bias": test_result["metrics_a"].get("bias", {}).get("value")
                }
        
        if "metrics_b" in test_result:
            prompt_b_idx = next(
                (i for i, p in enumerate(self.registry["prompts"]) 
                 if p["id"] == test_result["prompt_b"]),
                None
            )
            if prompt_b_idx is not None and test_result["metrics_b"]:
                self.registry["prompts"][prompt_b_idx]["results"] = {
                    "f1": test_result["metrics_b"].get("f1", {}).get("value"),
                    "tnr": test_result["metrics_b"].get("tnr", {}).get("value"),
                    "bias": test_result["metrics_b"].get("bias", {}).get("value")
                }
        
        self._save_registry()

    def set_production_prompt(self, prompt_id: str) -> None:
        """
        Set prompt as current production version.
        
        Args:
            prompt_id: ID of prompt to promote to production
        """
        if prompt_id not in [p["id"] for p in self.registry["prompts"]]:
            raise ValueError(f"Prompt {prompt_id} not found in registry")
        
        self.registry["current_production"] = prompt_id
        self.registry["last_updated"] = datetime.now().isoformat()
        self._save_registry()

    def get_production_prompt(self) -> str:
        """Get current production prompt ID."""
        return self.registry.get("current_production", "v1_zero_shot")
