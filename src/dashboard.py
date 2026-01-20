"""
Dashboard: Streamlit dashboard for visualization and monitoring.

Displays:
- Top-level metrics cards (F1, TNR, Bias, Consistency)
- Metrics over time chart
- A/B test results tab
- Drift detection alerts tab
- Failure mode breakdown tab
"""

import json
import glob
from pathlib import Path
from typing import Dict, List

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from metrics import MetricsCalculator
from monitoring import DriftMonitor
from regression import RegressionTester

# Human-friendly labels and brief definitions for metrics
METRIC_INFO = {
    "f1": {
        "label": "F1 Score",
        "help": "Balanced correctness: harmonic mean of precision and recall. Target ≥ 0.75",
    },
    "tnr": {
        "label": "TNR (Specificity)",
        "help": "True Negative Rate: avoids over-flagging grounded claims. Target ≥ 0.65",
    },
    "bias": {
        "label": "|Bias|",
        "help": "Absolute difference between false positive and false negative rates. Target ≤ 0.15",
    },
    "accuracy": {
        "label": "Accuracy",
        "help": "Overall proportion of correct predictions",
    },
    "precision": {
        "label": "Precision",
        "help": "Of the flagged hallucinations, how often are they correct",
    },
    "recall": {
        "label": "Recall",
        "help": "Of all hallucinations, how many were correctly flagged",
    },
    "cohens_kappa": {
        "label": "Cohen's Kappa",
        "help": "Agreement beyond chance between predictions and ground truth",
    },
}


def load_results(directory: str = "data/results") -> Dict[str, List[Dict]]:
    """Load all results files from directory."""
    results = {}
    for filepath in sorted(glob.glob(f"{directory}/*.json"), reverse=True):
        filename = Path(filepath).stem
        with open(filepath, "r") as f:
            results[filename] = json.load(f)
    return results


def calculate_metrics_from_results(results: List[Dict]) -> Dict[str, float]:
    """Calculate all metrics from results."""
    y_true = [r.get("ground_truth") for r in results if "ground_truth" in r]
    y_pred = [r.get("prediction") for r in results if "prediction" in r]
    
    if not y_true or not y_pred:
        return {}
    
    calc = MetricsCalculator(y_true, y_pred)
    metrics = calc.all_metrics()
    
    return {name: metric.value for name, metric in metrics.items()}


def main():
    """Main dashboard application."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit not installed. Run: pip install streamlit plotly")
        return
    
    st.set_page_config(page_title="Hallucination Detection Evaluation", layout="wide")
    st.title("🔍 Hallucination Detection Evaluation Dashboard")
    
    # Load results
    all_results = load_results()
    if not all_results:
        st.warning("No results found. Run evaluations first.")
        return
    
    # Sidebar: Select results file
    result_files = list(all_results.keys())
    selected_result = st.sidebar.selectbox("Select Results File", result_files)
    results = all_results[selected_result]
    
    # Calculate metrics
    metrics = calculate_metrics_from_results(results)
    
    # ===== TAB 1: METRICS OVERVIEW =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics Overview",
        "📈 Metrics History",
        "🆚 A/B Tests",
        "⚠️ Drift & Regression"
    ])
    
    with tab1:
        st.header("Primary Metrics")
        
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            f1_value = metrics.get("f1", 0.0)
            f1_color = "🟢" if f1_value >= 0.75 else "🟡" if f1_value >= 0.60 else "🔴"
            st.metric(METRIC_INFO["f1"]["label"], f"{f1_value:.3f}", f"{f1_color}")
            st.caption(METRIC_INFO["f1"]["help"])  # tooltip-style help
        
        with col2:
            tnr_value = metrics.get("tnr", 0.0)
            tnr_color = "🟢" if tnr_value >= 0.65 else "🟡" if tnr_value >= 0.50 else "🔴"
            st.metric(METRIC_INFO["tnr"]["label"], f"{tnr_value:.3f}", f"{tnr_color}")
            st.caption(METRIC_INFO["tnr"]["help"])  # tooltip-style help
        
        with col3:
            bias_value = abs(metrics.get("bias", 0.0))
            bias_color = "🟢" if bias_value <= 0.15 else "🟡" if bias_value <= 0.25 else "🔴"
            st.metric(METRIC_INFO["bias"]["label"], f"{bias_value:.3f}", f"{bias_color}")
            st.caption(METRIC_INFO["bias"]["help"])  # tooltip-style help
        
        with col4:
            accuracy_value = metrics.get("accuracy", 0.0)
            st.metric(METRIC_INFO["accuracy"]["label"], f"{accuracy_value:.3f}")
            st.caption(METRIC_INFO["accuracy"]["help"])  # tooltip-style help
        
        # Secondary metrics
        st.subheader("Secondary Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(METRIC_INFO["precision"]["label"], f"{metrics.get('precision', 0.0):.3f}")
            st.caption(METRIC_INFO["precision"]["help"])  # tooltip-style help
        
        with col2:
            st.metric(METRIC_INFO["recall"]["label"], f"{metrics.get('recall', 0.0):.3f}")
            st.caption(METRIC_INFO["recall"]["help"])  # tooltip-style help
        
        with col3:
            st.metric(METRIC_INFO["cohens_kappa"]["label"], f"{metrics.get('cohens_kappa', 0.0):.3f}")
            st.caption(METRIC_INFO["cohens_kappa"]["help"])  # tooltip-style help

        # Optional: quick reference of metric definitions
        with st.expander("ℹ️ Metric definitions", expanded=False):
            for key, info in METRIC_INFO.items():
                st.markdown(f"- **{info['label']}**: {info['help']}")
        
        # Results summary
        st.subheader("Results Summary")
        total_cases = len(results)
        correct_cases = sum(1 for r in results if r.get("correct", False))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cases", total_cases)
        with col2:
            st.metric("Correct", correct_cases)
        with col3:
            st.metric("Error Rate", f"{(1 - correct_cases/total_cases):.1%}")
    
    with tab2:
        st.header("Metrics Over Time")
        
        # Load all result files and extract metrics
        timestamps = []
        metric_history = {m: [] for m in ["f1", "precision", "recall", "tnr", "accuracy"]}
        
        for filename in sorted(result_files):
            try:
                res = all_results[filename]
                metric_values = calculate_metrics_from_results(res)
                
                for metric_name in metric_history:
                    if metric_name in metric_values:
                        metric_history[metric_name].append(metric_values[metric_name])
                
                timestamps.append(filename.replace("results_", ""))
            except:
                pass
        
        if timestamps and any(metric_history.values()):
            df = pd.DataFrame(metric_history, index=timestamps)
            
            fig = go.Figure()
            for col in df.columns:
                label = METRIC_INFO.get(col, {}).get("label", col.upper())
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode='lines+markers',
                    name=label
                ))
            
            fig.update_layout(
                title="Metrics Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Metric Value",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need multiple result files to show history. Run evaluations over time.")
    
    with tab3:
        st.header("A/B Test Results")
        
        # Load prompt registry
        try:
            with open("prompts/prompt_registry.json", "r") as f:
                registry = json.load(f)
            
            ab_tests = registry.get("ab_tests", [])
            if ab_tests:
                for test in ab_tests:
                    with st.expander(f"Test: {test['test_id']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Prompt A**: {test['prompt_a']}")
                        with col2:
                            st.write(f"**Prompt B**: {test['prompt_b']}")
                        with col3:
                            winner = test.get("winner", "unknown")
                            winner_str = winner.upper() if winner else "UNKNOWN"
                            winner_char = "🏆" if winner in ["a", "b"] else "🤝"
                            st.write(f"**Winner**: {winner_char} {winner_str}")
                        
                        if "metrics" in test and test["metrics"]:
                            st.write(f"**F1 Improvement**: {test['metrics'].get('f1_improvement', 0):.1%}")
                            st.write(f"**TNR Improvement**: {test['metrics'].get('tnr_improvement', 0):.1%}")
            else:
                st.info("No A/B tests completed yet.")
        except FileNotFoundError:
            st.warning("Prompt registry not found.")
    
    with tab4:
        st.header("Drift Detection & Regression Testing")
        
        # Regression test
        st.subheader("Regression Test Results")
        try:
            with open("data/test_cases/regression.json", "r") as f:
                regression_cases = json.load(f)
            
            # Try to match regression cases in current results
            regression_results = [r for r in results if any(
                r.get("test_case_id", "").startswith("REG_") for _ in [1]
            )]
            
            if regression_results:
                tester = RegressionTester()
                passed, details = tester.run_regression_test(regression_results)
                
                status_color = "🟢" if passed else "🔴"
                st.subheader(f"Regression Test: {status_color} {'PASS' if passed else 'FAIL'}")
                
                for metric_name, metric_details in details.items():
                    metric_status = "✓" if metric_details["pass"] else "✗"
                    st.write(f"{metric_status} **{metric_name.upper()}**: {metric_details['value']:.3f} "
                           f"(threshold: {metric_details['threshold']})")
            else:
                st.info("Run evaluation on regression test set to see results.")
        except FileNotFoundError:
            st.warning("Regression test set not found.")


if __name__ == "__main__":
    main()
