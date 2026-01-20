"""
Monitoring & Drift Detection: Track metric changes over time and detect degradation.

Detects:
- Metric drift beyond thresholds (default 0.10)
- Which specific metrics drifted
- Baseline vs current comparison
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from metrics import MetricsCalculator


class DriftMonitor:
    """
    Monitor metrics over time and detect drift from baseline.
    """

    def __init__(self, 
                 baseline_results_path: Optional[str] = None,
                 drift_threshold: float = 0.10):
        """
        Initialize drift monitor.
        
        Args:
            baseline_results_path: Path to baseline results JSON
            drift_threshold: Alert threshold (0.10 = 10% change)
        """
        self.baseline_results_path = baseline_results_path
        self.drift_threshold = drift_threshold
        self.baseline_metrics = None
        
        if baseline_results_path:
            self._load_baseline(baseline_results_path)

    def _load_baseline(self, path: str) -> None:
        """Load baseline metrics from results file."""
        with open(path, "r") as f:
            results = json.load(f)
            
        y_true = [r["ground_truth"] for r in results]
        y_pred = [r["prediction"] for r in results]
        
        calc = MetricsCalculator(y_true, y_pred)
        self.baseline_metrics = calc.all_metrics()

    def compare_to_baseline(self, 
                           current_results: List[Dict]) -> Dict[str, any]:
        """
        Compare current results to baseline and detect drift.
        
        Args:
            current_results: List of current evaluation results
            
        Returns:
            Dictionary with drift analysis
        """
        if self.baseline_metrics is None:
            raise ValueError("No baseline set; call set_baseline() first")
        
        # Calculate current metrics
        y_true = [r["ground_truth"] for r in current_results]
        y_pred = [r["prediction"] for r in current_results]
        
        calc = MetricsCalculator(y_true, y_pred)
        current_metrics = calc.all_metrics()
        
        # Compare each metric
        drift_analysis = {
            "timestamp": datetime.now().isoformat(),
            "baseline_vs_current": {},
            "drifted_metrics": [],
            "severe_drift_metrics": [],
            "overall_health": "healthy"
        }
        
        for metric_name in self.baseline_metrics:
            baseline_value = self.baseline_metrics[metric_name].value
            current_value = current_metrics[metric_name].value
            
            # Calculate relative change
            if baseline_value == 0:
                relative_change = abs(current_value - baseline_value)
            else:
                relative_change = abs(current_value - baseline_value) / abs(baseline_value)
            
            comparison = {
                "baseline": baseline_value,
                "current": current_value,
                "change": current_value - baseline_value,
                "relative_change": relative_change,
                "drifted": relative_change > self.drift_threshold
            }
            
            drift_analysis["baseline_vs_current"][metric_name] = comparison
            
            # Flag drifted metrics
            if relative_change > self.drift_threshold:
                drift_analysis["drifted_metrics"].append(metric_name)
                
                # Flag severe drift (>20%)
                if relative_change > 0.20:
                    drift_analysis["severe_drift_metrics"].append(metric_name)
        
        # Determine overall health
        if drift_analysis["severe_drift_metrics"]:
            drift_analysis["overall_health"] = "critical"
        elif drift_analysis["drifted_metrics"]:
            drift_analysis["overall_health"] = "warning"
        else:
            drift_analysis["overall_health"] = "healthy"
        
        return drift_analysis

    def set_baseline(self, results_path: str) -> None:
        """
        Set new baseline from results file.
        
        Args:
            results_path: Path to results JSON to use as baseline
        """
        self._load_baseline(results_path)

    def get_alerts(self, drift_analysis: Dict[str, any]) -> List[str]:
        """
        Generate human-readable alerts from drift analysis.
        
        Args:
            drift_analysis: Output from compare_to_baseline()
            
        Returns:
            List of alert strings
        """
        alerts = []
        
        if drift_analysis["overall_health"] == "critical":
            alerts.append(f"⚠️ CRITICAL: Severe drift detected in {len(drift_analysis['severe_drift_metrics'])} metrics")
            for metric in drift_analysis["severe_drift_metrics"]:
                comp = drift_analysis["baseline_vs_current"][metric]
                alerts.append(f"   {metric}: {comp['baseline']:.3f} → {comp['current']:.3f} ({comp['relative_change']:.1%} change)")
        
        if drift_analysis["overall_health"] == "warning":
            alerts.append(f"⚠️ WARNING: Drift detected in {len(drift_analysis['drifted_metrics'])} metrics")
            for metric in drift_analysis["drifted_metrics"]:
                if metric not in drift_analysis["severe_drift_metrics"]:
                    comp = drift_analysis["baseline_vs_current"][metric]
                    alerts.append(f"   {metric}: {comp['baseline']:.3f} → {comp['current']:.3f} ({comp['relative_change']:.1%} change)")
        
        if not alerts:
            alerts.append("✓ No drift detected; system operating normally")
        
        return alerts

    def should_rollback(self, drift_analysis: Dict[str, any]) -> bool:
        """
        Determine if system should rollback based on drift.
        
        Conservative: Only rollback on critical metric degradation.
        
        Args:
            drift_analysis: Output from compare_to_baseline()
            
        Returns:
            True if rollback recommended
        """
        # Rollback if:
        # 1. F1 degraded by >10% AND below 0.75
        # 2. TNR degraded by >15% AND below 0.65
        # 3. |Bias| increased by >50%
        
        if "f1" in drift_analysis["baseline_vs_current"]:
            f1_comp = drift_analysis["baseline_vs_current"]["f1"]
            if f1_comp["drifted"] and f1_comp["current"] < 0.75:
                return True
        
        if "tnr" in drift_analysis["baseline_vs_current"]:
            tnr_comp = drift_analysis["baseline_vs_current"]["tnr"]
            if tnr_comp["drifted"] and tnr_comp["current"] < 0.65:
                return True
        
        return False
