"""
DEPRECATED: This is a minimal evaluation runner UI.
Use dashboard_v2.py instead, which includes evaluation functionality.

To run the current dashboard:
    python -m streamlit run src/dashboard_v2.py

This file is kept for reference only.
"""

import streamlit as st
from src.evaluator import Evaluator
from src.metrics import MetricsCalculator
import os
import datetime

# --- UI ---
st.warning("This UI is deprecated. Please use dashboard_v2.py instead.")
st.title("LLM Hallucination Evaluation Runner (Legacy)")

# Discover available prompts from the prompts/ folder
def get_prompt_choices():
    prompt_dir = os.path.join(os.path.dirname(__file__), '../prompts')
    files = [f for f in os.listdir(prompt_dir) if f.startswith('v') and f.endswith('.txt')]
    return [f[:-4] for f in files]  # strip .txt

prompt_choices = get_prompt_choices()
prompt_id = st.selectbox("Select prompt version to evaluate:", prompt_choices)
run_name = st.text_input("Run name (for results file):", value=f"{prompt_id}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

if st.button("Run Evaluation"):
    st.info(f"Running evaluation for {prompt_id} ... this may take a few minutes.")
    evaluator = Evaluator()
    test_case_ids = [c for c in evaluator.list_test_cases() if not c.startswith("REG_")]
    results = evaluator.evaluate_batch(prompt_id=prompt_id, test_case_ids=test_case_ids)
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"] for r in results]
    calc = MetricsCalculator(y_true, y_pred)
    metrics = calc.all_metrics()
    evaluator.save_results(results, run_name=run_name)
    st.success(f"Evaluation complete! Results saved as {run_name}.")
    st.subheader("Metrics:")
    for name, metric in metrics.items():
        st.write(f"**{name}**: {metric.value:.3f} - {metric.interpretation}")
    is_ready, details = calc.production_ready()
    st.write(f"**Production ready:** {is_ready}")
    st.write(details)

st.markdown("---")
st.markdown("After running, refresh the main dashboard to compare all runs.")
