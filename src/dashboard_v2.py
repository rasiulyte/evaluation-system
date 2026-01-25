import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
import glob
from pathlib import Path

# Import database abstraction (supports SQLite and PostgreSQL)
import sys
sys.path.insert(0, "src")
from database import db


def load_metrics() -> pd.DataFrame:
    """Load metrics from database."""
    try:
        metrics = db.get_all_metrics()
        if metrics:
            return pd.DataFrame(metrics)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_test_case_definitions() -> dict:
    """Load full test case definitions from ground_truth.json and regression.json."""
    test_case_defs = {}

    for filepath in ["data/test_cases/ground_truth.json", "data/test_cases/regression.json"]:
        try:
            with open(filepath, "r") as f:
                cases = json.load(f)
                for case in cases:
                    test_case_defs[case.get("id", case.get("test_case_id", ""))] = case
        except Exception:
            pass

    return test_case_defs


def load_daily_runs() -> pd.DataFrame:
    """Load daily run history from database (derived from metrics)."""
    try:
        metrics = db.get_all_metrics()
        if not metrics:
            return pd.DataFrame()
        # Derive run info from metrics
        df = pd.DataFrame(metrics)
        runs = df.groupby('run_id').agg({
            'timestamp': 'first',
            'scenario': 'nunique'
        }).reset_index()
        runs.columns = ['run_id', 'timestamp', 'scenarios_run']
        runs['run_date'] = runs['timestamp'].str[:10]
        return runs.sort_values('timestamp', ascending=False)
    except Exception:
        return pd.DataFrame()


def get_metrics_comparison(run_id_1: str, run_id_2: str) -> pd.DataFrame:
    """Compare metrics between two runs."""
    try:
        # Use database abstraction - returns list of dicts
        # For now, load all metrics and filter in pandas
        metrics = db.get_all_metrics()
        if not metrics:
            return pd.DataFrame()

        df = pd.DataFrame(metrics)
        m1 = df[df['run_id'] == run_id_1][['scenario', 'metric_name', 'metric_value']].copy()
        m2 = df[df['run_id'] == run_id_2][['scenario', 'metric_name', 'metric_value']].copy()

        merged = m1.merge(m2, on=['scenario', 'metric_name'], suffixes=('_1', '_2'))
        merged['run1_value'] = merged['metric_value_1']
        merged['run2_value'] = merged['metric_value_2']
        merged['delta'] = merged['run2_value'] - merged['run1_value']
        merged['delta_pct'] = merged.apply(
            lambda r: ((r['run2_value'] - r['run1_value']) / r['run1_value'] * 100) if r['run1_value'] > 0 else 0,
            axis=1
        )
        return merged[['scenario', 'metric_name', 'run1_value', 'run2_value', 'delta', 'delta_pct']]
    except Exception as e:
        print(f"Comparison error: {e}")
        return pd.DataFrame()


def get_runs_with_metrics() -> list:
    """Get list of run_ids that have metrics data."""
    try:
        return db.get_run_ids()
    except Exception:
        return []


def render_test_case_detail(case_id: str, test_case_defs: dict, result_data: dict = None):
    """Render a detailed view of a test case in a dialog/expander."""

    case_def = test_case_defs.get(case_id, {})

    if not case_def:
        st.warning(f"No definition found for test case: {case_id}")
        return

    # Header with status
    label = case_def.get("label", "unknown")
    label_color = "🔴" if label == "hallucination" else "🟢"

    st.markdown(f"### {label_color} Test Case: `{case_id}`")

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Ground Truth:** `{label}`")
    with col2:
        difficulty = case_def.get("difficulty", "unknown")
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
        st.markdown(f"**Difficulty:** {diff_icon} {difficulty}")
    with col3:
        failure_mode = case_def.get("failure_mode", case_id.split("_")[0] if "_" in case_id else "unknown")
        st.markdown(f"**Failure Mode:** `{failure_mode}`")

    # Context
    st.markdown("**📄 Context (Source Material):**")
    context = case_def.get("context", "No context available")
    st.text_area("Context", context, height=150, disabled=True, key=f"ctx_{case_id}")

    # Response being evaluated
    st.markdown("**💬 Response Being Evaluated:**")
    response = case_def.get("response", "No response available")
    st.text_area("Response", response, height=100, disabled=True, key=f"resp_{case_id}")

    # Hallucination span (if present)
    if case_def.get("hallucination_span"):
        st.markdown("**🎯 Hallucination Span:**")
        st.error(f'"{case_def["hallucination_span"]}"')

    # Reasoning
    if case_def.get("reasoning"):
        st.markdown("**🧠 Why This is Labeled as Such:**")
        st.info(case_def["reasoning"])

    # If we have result data, show prediction vs ground truth
    if result_data:
        st.markdown("---")
        st.markdown("**📊 Evaluation Result:**")

        pred = result_data.get("prediction", "unknown")
        correct = result_data.get("correct", False)
        confidence = result_data.get("confidence", 0)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Prediction", pred)
        with res_col2:
            st.metric("Correct?", "✅ Yes" if correct else "❌ No")
        with res_col3:
            st.metric("Confidence", f"{confidence:.2f}" if confidence else "N/A")

        # LLM output if available
        if result_data.get("llm_output"):
            with st.expander("🤖 Full LLM Output"):
                st.code(result_data["llm_output"], language=None)

METRIC_LABELS = {
    "f1": "F1 Score",
    "tnr": "TNR (Specificity)",
    "bias": "|Bias|",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "cohens_kappa": "Cohen's Kappa",
}


def get_alert_explanation(metric: str, value: float, status: str, thresh_min, thresh_max) -> dict:
    """Generate detailed explanation for a metric alert."""

    explanations = {
        "f1": {
            "title": f"🔴 F1 Score Critical: {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"The F1 score dropped to {value:.3f}, which is below the minimum threshold of {thresh_min}. This means the balance between precision and recall is poor.",
            "why": "F1 is your primary metric for hallucination detection accuracy. A low F1 means either you're missing real hallucinations (low recall) OR flagging good content as hallucinations (low precision). Users will either see undetected hallucinations or get frustrated by false alarms.",
            "fix": "1. Check precision vs recall separately to identify the issue\n   2. If precision is low → tighten detection criteria, add more specific examples\n   3. If recall is low → broaden detection criteria, add examples of missed hallucinations\n   4. Consider upgrading to a more sophisticated prompt (v3 chain-of-thought or v5 structured)"
        },
        "tnr": {
            "title": f"🟠 True Negative Rate Low: {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"TNR (Specificity) is {value:.3f}, below the {thresh_min} threshold. The system is incorrectly flagging legitimate, grounded content as hallucinations.",
            "why": "Low TNR means too many false positives. Users will lose trust when valid responses get flagged. In production, this leads to alert fatigue and users ignoring warnings altogether.",
            "fix": "1. Review false positive cases - what grounded content is being flagged?\n   2. Add examples of valid inferences (FM4) and verbatim grounded content (FM5) to your prompt\n   3. Adjust prompt to be less aggressive - emphasize 'only flag clear hallucinations'\n   4. Check if the model is being too literal in interpretation"
        },
        "bias": {
            "title": f"🟡 Prediction Bias Detected: {value:.3f} (threshold: |bias| ≤{thresh_max})",
            "what": f"Bias is {value:.3f}, exceeding the ±{thresh_max} threshold. " + ("The system over-predicts hallucinations (too many false positives)." if value > 0 else "The system under-predicts hallucinations (too many false negatives)."),
            "why": "Systematic bias means the model has a tendency in one direction. " + ("Positive bias frustrates users with false alarms." if value > 0 else "Negative bias lets real hallucinations slip through undetected."),
            "fix": ("1. Add more examples of grounded content to balance the prompt\n   2. Explicitly tell the model 'when in doubt, mark as grounded'\n   3. Review FM4 (Valid Inference) and FM5 (Verbatim) test cases" if value > 0 else "1. Add more examples of subtle hallucinations (FM3, FM6, FM7)\n   2. Explicitly tell the model to be more vigilant\n   3. Review FM6 (Fluent Hallucination) cases - well-written doesn't mean correct")
        },
        "accuracy": {
            "title": f"🔴 Accuracy Below Threshold: {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"Overall accuracy is {value:.3f}, below the {thresh_min} threshold. The system is making too many errors overall.",
            "why": "Low accuracy means the fundamental detection capability is compromised. Both hallucinations and grounded content are being misclassified.",
            "fix": "1. This usually indicates a fundamental prompt issue - consider a major revision\n   2. Check if test cases are properly labeled\n   3. Review the most confident wrong predictions - what's the model misunderstanding?\n   4. Consider using a more capable base model"
        },
        "precision": {
            "title": f"🟠 Precision Low: {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"Precision is {value:.3f}. When the system flags something as a hallucination, it's wrong {(1-value)*100:.0f}% of the time.",
            "why": "Low precision erodes user trust. Every false alarm teaches users to ignore warnings.",
            "fix": "1. Make detection criteria more specific\n   2. Add examples showing what is NOT a hallucination\n   3. Require higher confidence before flagging"
        },
        "recall": {
            "title": f"🟠 Recall Low: {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"Recall is {value:.3f}. The system is missing {(1-value)*100:.0f}% of actual hallucinations.",
            "why": "Low recall means hallucinations slip through undetected. Users see AI-generated misinformation without warning.",
            "fix": "1. Broaden detection criteria\n   2. Add examples of subtle hallucinations (FM3, FM6, FM7)\n   3. Lower the confidence threshold for flagging"
        },
        "cohens_kappa": {
            "title": f"🟡 Consistency Issue: Cohen's Kappa = {value:.3f} (threshold: ≥{thresh_min})",
            "what": f"Cohen's Kappa is {value:.3f}, indicating inconsistent predictions across runs or poor agreement with ground truth.",
            "why": "Low kappa means unreliable results. The same input might get different classifications, making the system unpredictable.",
            "fix": "1. Reduce temperature in model settings for more deterministic outputs\n   2. Use structured output format (v5) for consistent parsing\n   3. Add more explicit decision criteria to the prompt"
        }
    }

    # Default fallback for unknown metrics
    default = {
        "title": f"⚠️ {metric.upper()} Alert: {value:.3f} ({status})",
        "what": f"The metric {metric} has value {value:.3f} which triggered a {status} status.",
        "why": "This metric is outside expected thresholds and may indicate a problem with model performance.",
        "fix": "1. Review the metric definition in docs/METRICS.md\n   2. Compare with previous runs to identify when the issue started\n   3. Check recent changes to prompts or test data"
    }

    return explanations.get(metric, default)


def render_run_evaluation_page():
    """Render the run evaluation page with UI controls."""
    st.header("🚀 Run Evaluation")

    # Initialize session state for persisting results
    if 'eval_summary' not in st.session_state:
        st.session_state.eval_summary = None

    # Load test case definitions
    test_case_defs = load_test_case_definitions()

    # Load config
    try:
        import yaml
        with open("config/settings.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Could not load config: {e}")
        return

    # Available scenarios and prompts
    scenarios_config = config.get("scenarios", {})
    available_prompts = []
    for f in glob.glob("prompts/*.txt"):
        available_prompts.append(Path(f).stem)

    # Get all test cases grouped by failure mode
    all_cases = list(test_case_defs.keys())
    train_cases = [c for c in all_cases if not c.startswith("REG_")]
    reg_cases = [c for c in all_cases if c.startswith("REG_")]

    # Group by failure mode
    fm_groups = {}
    for case_id in train_cases:
        fm = case_id.split("_")[0] if "_" in case_id else "Other"
        if fm not in fm_groups:
            fm_groups[fm] = []
        fm_groups[fm].append(case_id)

    # --- Configuration Section ---
    st.subheader("⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Scenario selection
        st.markdown("**Select Scenarios to Run:**")
        selected_scenarios = []
        for scenario_name, scenario_config in scenarios_config.items():
            default_enabled = scenario_config.get("enabled", False)
            if st.checkbox(f"{scenario_name}", value=default_enabled, key=f"scenario_{scenario_name}"):
                selected_scenarios.append(scenario_name)

    with col2:
        # Model and settings
        st.markdown("**Model Settings:**")
        model = st.selectbox("Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
        sample_size = st.slider("Sample Size (per scenario)", min_value=5, max_value=100, value=20)

    # --- Test Cases Section ---
    st.subheader("📋 Test Cases to Evaluate")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["By Failure Mode", "Individual Selection", "Quick Select"])

    with tab1:
        st.markdown("**Select Failure Modes:**")
        selected_fms = []
        cols = st.columns(4)
        fm_descriptions = {
            "FM1": "Factual Addition",
            "FM2": "Fabrication",
            "FM3": "Subtle Distortion",
            "FM4": "Valid Inference",
            "FM5": "Verbatim Grounded",
            "FM6": "Fluent Hallucination",
            "FM7": "Partial Grounding",
        }
        for i, (fm, cases) in enumerate(sorted(fm_groups.items())):
            with cols[i % 4]:
                desc = fm_descriptions.get(fm, fm)
                if st.checkbox(f"{fm} ({len(cases)})", value=True, key=f"fm_{fm}", help=desc):
                    selected_fms.append(fm)

        # Get cases from selected failure modes
        fm_selected_cases = []
        for fm in selected_fms:
            fm_selected_cases.extend(fm_groups.get(fm, []))

    with tab2:
        st.markdown("**Select Individual Test Cases:**")
        individual_cases = st.multiselect(
            "Choose specific test cases",
            options=train_cases,
            default=[],
            key="individual_cases"
        )

    with tab3:
        st.markdown("**Quick Selection:**")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            select_all = st.button("✅ Select All", key="select_all")
        with quick_col2:
            select_none = st.button("❌ Clear All", key="select_none")
        with quick_col3:
            include_regression = st.checkbox("Include Regression Cases (REG_*)", value=False)

    # Determine final test case list
    if individual_cases:
        final_cases = individual_cases
    else:
        final_cases = fm_selected_cases

    if include_regression:
        final_cases = final_cases + reg_cases

    # Apply sample size limit with random sampling
    import random
    import time

    if len(final_cases) > sample_size:
        random.seed(time.time())
        sampled_cases = random.sample(final_cases, sample_size)
        # Shuffle to ensure display order is random (not grouped by FM)
        random.shuffle(sampled_cases)
        will_sample = True
    else:
        sampled_cases = final_cases.copy()
        random.seed(time.time())
        random.shuffle(sampled_cases)  # Shuffle even if not sampling
        will_sample = False

    # --- Preview Section ---
    if will_sample:
        st.subheader(f"👁️ Preview: {sample_size} Cases (randomly sampled from {len(final_cases)} available)")
        st.info(f"📊 **{len(final_cases)}** cases available → **{sample_size}** will be randomly selected per scenario")
    else:
        st.subheader(f"👁️ Preview: {len(final_cases)} Test Cases")

    if final_cases:
        # Show failure mode distribution in the sample
        fm_distribution = {}
        for case_id in sampled_cases:
            fm = case_id.split("_")[0] if "_" in case_id else "Other"
            fm_distribution[fm] = fm_distribution.get(fm, 0) + 1

        # Display distribution summary
        dist_str = " | ".join([f"**{fm}**: {count}" for fm, count in sorted(fm_distribution.items())])
        st.markdown(f"📊 **Sample Distribution:** {dist_str}")

        # Show the sampled cases that will actually run
        preview_data = []
        for case_id in sampled_cases:
            case_def = test_case_defs.get(case_id, {})
            preview_data.append({
                "ID": case_id,
                "Label": case_def.get("label", "?"),
                "Difficulty": case_def.get("difficulty", "?"),
                "Failure Mode": case_id.split("_")[0] if "_" in case_id else "?",
                "Context Preview": (case_def.get("context", "")[:80] + "...") if case_def.get("context") else "N/A"
            })

        st.dataframe(
            pd.DataFrame(preview_data),
            use_container_width=True,
            height=400  # Scrollable table
        )

        if will_sample:
            st.caption(f"⚠️ Note: Actual cases will be randomly re-sampled when evaluation runs. This preview shows one possible sample.")
    else:
        st.warning("No test cases selected!")

    # --- Run Section ---
    st.subheader("🎯 Run Evaluation")

    # Summary before run
    st.markdown(f"""
    **Run Summary:**
    - **Scenarios:** {len(selected_scenarios)} selected ({', '.join(selected_scenarios) if selected_scenarios else 'None'})
    - **Test Cases:** {len(final_cases)} selected (sample: {min(sample_size, len(final_cases))})
    - **Model:** {model}
    """)

    # Check for API key (load from .env)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment. Set it in .env file.")
    else:
        st.success("✅ API Key detected")

    # Password protection for evaluation (hashed for security)
    import hashlib

    def hash_password(password: str) -> str:
        """Hash password with SHA-256 and salt."""
        salt = "eval_system_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()

    st.markdown("**🔐 Authorization:**")
    eval_password = st.text_input("Enter password to run evaluation", type="password", key="eval_password")

    # Stored hash (default is hash of "1978") - set EVAL_PASSWORD_HASH in .env for custom
    # To generate a new hash: python -c "import hashlib; print(hashlib.sha256(('YOUR_PASSWORD' + 'eval_system_salt_2024').encode()).hexdigest())"
    default_hash = "a]59e417de5ef9564e8c9b9e0a5a9d8a93c5b8c5d6e7f8a9b0c1d2e3f4a5b6c7d8"  # hash of "1978"
    stored_hash = os.getenv("EVAL_PASSWORD_HASH", hash_password("1978"))

    password_valid = False
    if eval_password:
        input_hash = hash_password(eval_password)
        password_valid = input_hash == stored_hash

    if eval_password and not password_valid:
        st.error("❌ Incorrect password")
    elif password_valid:
        st.success("✅ Password accepted")

    # Run buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        dry_run = st.button("🔍 Dry Run (Preview)", disabled=not selected_scenarios, key="dry_run")

    with col2:
        run_eval = st.button("▶️ Start Evaluation", type="primary",
                             disabled=not (selected_scenarios and final_cases and has_api_key and password_valid),
                             key="run_eval")

    with col3:
        if st.button("🔄 Reset Results", key="reset"):
            st.session_state.eval_summary = None
            st.rerun()

    # Handle dry run
    if dry_run:
        st.info("**Dry Run Preview:**")
        for scenario in selected_scenarios:
            scenario_config = scenarios_config.get(scenario, {})
            prompt = scenario_config.get("prompt_version", "v1_zero_shot")
            st.markdown(f"- **{scenario}**: prompt=`{prompt}`, cases={min(sample_size, len(final_cases))}")

    # Handle actual run
    if run_eval:
        st.session_state.eval_summary = None  # Clear previous results
        st.markdown("---")

        with st.status(f"🚀 Running evaluation...", expanded=True) as status:
            st.write(f"**Scenarios:** {', '.join(selected_scenarios)}")
            st.write(f"**Model:** {model}")
            st.write(f"**Sample size:** {sample_size}")

            # Estimate: ~0.5-1 second per API call
            total_calls = len(selected_scenarios) * sample_size
            est_seconds = total_calls * 0.75
            st.write(f"**Estimated API calls:** {total_calls} (~{est_seconds:.0f} seconds)")
            st.markdown("---")

            st.warning(f"⏳ **Evaluation in progress...** Making {total_calls} API calls.")

            try:
                import sys
                sys.path.insert(0, "src")
                from orchestrator import EvaluationOrchestrator

                orchestrator = EvaluationOrchestrator()

                # Run the evaluation
                summary = orchestrator.run_daily_evaluation(
                    scenarios=selected_scenarios,
                    model=model,
                    sample_size=sample_size
                )

                if summary:
                    # Store in session state for persistence
                    st.session_state.eval_summary = {
                        'run_id': summary.run_id,
                        'overall_status': summary.overall_status.value,
                        'scenarios_run': summary.scenarios_run,
                        'scenarios_passed': summary.scenarios_passed,
                        'scenarios_failed': summary.scenarios_failed,
                        'alerts': summary.alerts,
                        'hillclimb_suggestions': summary.hillclimb_suggestions
                    }
                    status.update(label="✅ Evaluation Complete!", state="complete", expanded=False)
                else:
                    status.update(label="❌ Evaluation Failed", state="error", expanded=True)
                    st.error("Evaluation returned no results. Check terminal for errors.")

            except Exception as e:
                status.update(label="❌ Evaluation Failed", state="error", expanded=True)
                st.error(f"Error: {e}")
                st.exception(e)

    # Display results from session state (persists after button click)
    if st.session_state.eval_summary:
        summary = st.session_state.eval_summary
        st.balloons()  # Celebration!

        st.markdown("---")
        st.header("✅ Evaluation Complete!")

        # Big success banner
        status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(summary['overall_status'], "⚪")
        st.success(f"""
        **Run ID:** `{summary['run_id']}`

        **Status:** {status_icon} {summary['overall_status'].upper()}

        **Results:** {summary['scenarios_passed']}/{summary['scenarios_run']} scenarios passed
        """)

        # Results metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scenarios Run", summary['scenarios_run'])
        with col2:
            st.metric("Passed", summary['scenarios_passed'], delta=None)
        with col3:
            st.metric("Failed", summary['scenarios_failed'], delta=None, delta_color="inverse")
        with col4:
            st.metric("Status", summary['overall_status'].upper())

        # Alerts
        if summary['alerts']:
            st.subheader("⚠️ Alerts")
            for alert in summary['alerts']:
                st.warning(alert)

        # Suggestions
        if summary['hillclimb_suggestions']:
            st.subheader("💡 Suggestions")
            for suggestion in summary['hillclimb_suggestions']:
                st.info(suggestion)

        # Next steps
        st.subheader("📍 Next Steps")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **View Results:**
            - Go to **📅 Daily Runs** to see history
            - Go to **🔄 Compare Runs** to compare with previous
            """)
        with col2:
            st.markdown("""
            **Run Again:**
            - Adjust scenarios/settings above
            - Click **▶️ Start Evaluation** again
            """)

        # Clear results button
        if st.button("🗑️ Clear Results", key="clear_results"):
            st.session_state.eval_summary = None
            st.rerun()


def render_daily_runs_page(df: pd.DataFrame):
    """Render the daily runs history page."""
    st.header("📅 Daily Evaluation Runs")

    # Get runs from metrics table (more reliable than daily_runs)
    if df.empty or "run_id" not in df.columns:
        st.info("No runs recorded yet. Use **🚀 Run Evaluation** to generate data.")
        return

    # Build runs summary from metrics data
    runs_summary = []
    for run_id in df["run_id"].unique():
        run_df = df[df["run_id"] == run_id]
        timestamp = run_df["timestamp"].iloc[0] if not run_df.empty else ""
        run_date = timestamp[:10] if timestamp else ""

        # Check health status based on metrics
        f1_val = run_df[run_df["metric_name"] == "f1"]["metric_value"].values
        tnr_val = run_df[run_df["metric_name"] == "tnr"]["metric_value"].values
        bias_val = run_df[run_df["metric_name"] == "bias"]["metric_value"].values

        f1 = f1_val[0] if len(f1_val) > 0 else 0
        tnr = tnr_val[0] if len(tnr_val) > 0 else 0
        bias = bias_val[0] if len(bias_val) > 0 else 0

        # Determine status
        if f1 >= 0.75 and tnr >= 0.65 and abs(bias) <= 0.15:
            status = "healthy"
        elif f1 < 0.6 or tnr < 0.5:
            status = "critical"
        else:
            status = "warning"

        scenarios = run_df["scenario"].nunique()

        runs_summary.append({
            "run_id": run_id,
            "run_date": run_date,
            "timestamp": timestamp,
            "scenarios": scenarios,
            "f1": f1,
            "tnr": tnr,
            "bias": bias,
            "status": status
        })

    runs_df = pd.DataFrame(runs_summary).sort_values("timestamp", ascending=False)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", len(runs_df))
    with col2:
        healthy_runs = len(runs_df[runs_df["status"] == "healthy"])
        st.metric("Healthy Runs", healthy_runs)
    with col3:
        latest_status = runs_df.iloc[0]["status"] if not runs_df.empty else "N/A"
        status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(latest_status, "⚪")
        st.metric("Latest Status", f"{status_icon} {latest_status.upper()}")
    with col4:
        st.metric("Total Runs", f"{len(runs_df)} runs")

    # Runs history table
    st.subheader("📋 Run History")

    # Add status icons
    runs_df["status_icon"] = runs_df["status"].apply(
        lambda x: {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(x, "⚪")
    )

    st.dataframe(
        runs_df[["run_id", "run_date", "status_icon", "scenarios", "f1", "tnr", "bias", "status"]],
        use_container_width=True,
        column_config={
            "run_id": st.column_config.TextColumn("Run ID"),
            "run_date": st.column_config.TextColumn("Date"),
            "status_icon": st.column_config.TextColumn("", width="small"),
            "scenarios": st.column_config.NumberColumn("Scenarios"),
            "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            "tnr": st.column_config.NumberColumn("TNR", format="%.3f"),
            "bias": st.column_config.NumberColumn("Bias", format="%.3f"),
            "status": st.column_config.TextColumn("Status"),
        }
    )

    # Metrics trend over time
    st.subheader("📈 Metrics Trend Over Time")

    if "run_id" in df.columns:
        # Get unique runs in order
        run_order = df.groupby("run_id")["timestamp"].min().sort_values().index.tolist()

        if len(run_order) > 1:
            # Pivot for trend chart
            for metric_name in ["f1", "tnr", "bias"]:
                metric_df = df[df["metric_name"] == metric_name].copy()
                if not metric_df.empty:
                    fig = go.Figure()

                    for scenario in metric_df["scenario"].unique():
                        sdf = metric_df[metric_df["scenario"] == scenario].sort_values("timestamp")
                        fig.add_trace(go.Scatter(
                            x=sdf["timestamp"],
                            y=sdf["metric_value"],
                            mode="lines+markers",
                            name=scenario
                        ))

                    # Add threshold line
                    threshold = 0.75 if metric_name == "f1" else (0.65 if metric_name == "tnr" else 0.15)
                    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                                  annotation_text=f"Threshold: {threshold}")

                    fig.update_layout(
                        title=f"{METRIC_LABELS.get(metric_name, metric_name)} Over Time",
                        xaxis_title="Run Date",
                        yaxis_title="Value",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need multiple runs to show trends. Run the orchestrator daily to track progress.")

    # Alerts from latest run
    st.subheader("⚠️ Latest Run Alerts")
    if not runs_df.empty:
        latest_alerts = runs_df.iloc[0].get("alerts", "[]")
        try:
            alerts = json.loads(latest_alerts) if isinstance(latest_alerts, str) else latest_alerts
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("No alerts from latest run!")
        except:
            st.info("No alerts data available.")

    # How to run daily
    with st.expander("🔧 How to Run Daily Evaluations"):
        st.markdown("""
        **Manual Run:**
        ```bash
        python -m src.orchestrator
        ```

        **Schedule with Windows Task Scheduler:**
        1. Open Task Scheduler
        2. Create Basic Task → Name: "Daily LLM Evaluation"
        3. Trigger: Daily at your preferred time
        4. Action: Start a program
           - Program: `python`
           - Arguments: `-m src.orchestrator`
           - Start in: `C:\\path\\to\\evaluation-system`

        **Schedule with cron (Linux/Mac):**
        ```bash
        # Add to crontab (crontab -e)
        0 6 * * * cd /path/to/evaluation-system && python -m src.orchestrator
        ```
        """)


def render_compare_runs_page(df: pd.DataFrame):
    """Render the run comparison page."""
    st.header("🔄 Compare Runs")

    # Get runs that actually have metrics data
    run_ids = get_runs_with_metrics()

    if len(run_ids) < 2:
        st.warning(f"Need at least 2 runs with metrics data to compare. Currently have {len(run_ids)} run(s).")
        st.info("Run the orchestrator to generate more data:")
        st.code("python -m src.orchestrator", language="bash")
        return

    # Select runs to compare
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline Run")
        baseline_run = st.selectbox("Select baseline", run_ids, index=min(1, len(run_ids)-1), key="baseline")

    with col2:
        st.subheader("Compare Run")
        compare_run = st.selectbox("Select run to compare", run_ids, index=0, key="compare")

    if baseline_run == compare_run:
        st.warning("Please select different runs to compare.")
        return

    # Get comparison data
    comparison_df = get_metrics_comparison(baseline_run, compare_run)

    if comparison_df.empty:
        st.warning("Could not load comparison data.")
        return

    # Summary
    st.subheader("📊 Comparison Summary")

    improved = len(comparison_df[comparison_df["delta"] > 0])
    declined = len(comparison_df[comparison_df["delta"] < 0])
    unchanged = len(comparison_df[comparison_df["delta"] == 0])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Improved", f"📈 {improved}", delta=None)
    with col2:
        st.metric("Declined", f"📉 {declined}", delta=None)
    with col3:
        st.metric("Unchanged", f"➡️ {unchanged}", delta=None)

    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")

    # Format the dataframe for display
    comparison_df["change"] = comparison_df.apply(
        lambda row: f"{'📈' if row['delta'] > 0 else '📉' if row['delta'] < 0 else '➡️'} {row['delta']:+.3f} ({row['delta_pct']:+.1f}%)",
        axis=1
    )

    st.dataframe(
        comparison_df[["scenario", "metric_name", "run1_value", "run2_value", "change"]],
        use_container_width=True,
        column_config={
            "scenario": st.column_config.TextColumn("Scenario"),
            "metric_name": st.column_config.TextColumn("Metric"),
            "run1_value": st.column_config.NumberColumn(f"Baseline ({baseline_run[:15]}...)", format="%.3f"),
            "run2_value": st.column_config.NumberColumn(f"Compare ({compare_run[:15]}...)", format="%.3f"),
            "change": st.column_config.TextColumn("Change"),
        }
    )

    # Visual comparison chart
    st.subheader("📈 Visual Comparison")

    scenarios = comparison_df["scenario"].unique()
    selected_scenario = st.selectbox("Select scenario to visualize", scenarios)

    scenario_data = comparison_df[comparison_df["scenario"] == selected_scenario]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Baseline ({baseline_run[:10]})",
        x=scenario_data["metric_name"],
        y=scenario_data["run1_value"],
        marker_color="lightblue"
    ))
    fig.add_trace(go.Bar(
        name=f"Compare ({compare_run[:10]})",
        x=scenario_data["metric_name"],
        y=scenario_data["run2_value"],
        marker_color="darkblue"
    ))

    fig.update_layout(
        title=f"Metrics Comparison: {selected_scenario}",
        barmode="group",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("💡 Recommendations")

    significant_declines = comparison_df[(comparison_df["delta"] < -0.05)]
    if not significant_declines.empty:
        st.error("**Significant Declines Detected:**")
        for _, row in significant_declines.iterrows():
            st.markdown(f"- **{row['scenario']} - {row['metric_name']}**: Dropped from {row['run1_value']:.3f} to {row['run2_value']:.3f} ({row['delta_pct']:.1f}%)")
        st.markdown("""
        **Recommended Actions:**
        1. Check if test data changed between runs
        2. Review prompt modifications
        3. Check for model API changes or degradation
        4. Consider rolling back to baseline configuration
        """)
    else:
        st.success("No significant performance declines detected!")


def main():
    try:
        st.set_page_config(page_title="Evaluation Dashboard v2", layout="wide")
        st.title("📊 Evaluation Dashboard v2 (DB-powered)")

        df = load_metrics()

        # Sidebar: Navigation (always show)
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("View", ["🚀 Run Evaluation", "📈 Current Metrics", "📅 Daily Runs", "🔄 Compare Runs"])

        # Run Evaluation page works without data
        if page == "🚀 Run Evaluation":
            render_run_evaluation_page()
            return

        # Other pages need data
        if df.empty:
            st.warning("No metrics found in database. Use **🚀 Run Evaluation** to generate data.")
            return

        # Sidebar: Filters (only show for metrics pages)
        if page in ["📈 Current Metrics"]:
            st.sidebar.markdown("---")
            scenarios = sorted(df["scenario"].unique())
            scenario = st.sidebar.selectbox("Scenario", scenarios)
            metrics_list = [m for m in METRIC_LABELS.keys() if m in df["metric_name"].unique()]
            metric = st.sidebar.selectbox("Metric", metrics_list)

        # Route to different pages
        if page == "📅 Daily Runs":
            render_daily_runs_page(df)
            return
        elif page == "🔄 Compare Runs":
            render_compare_runs_page(df)
            return

        # Default: Current Metrics page

        # Filtered data
        sdf = df[df["scenario"] == scenario]
        mdf = sdf[sdf["metric_name"] == metric]

        # Metrics overview
        st.header(f"{METRIC_LABELS.get(metric, metric)} for {scenario}")
        st.metric(
            label=f"Latest {METRIC_LABELS.get(metric, metric)}",
            value=f"{mdf['metric_value'].iloc[-1]:.3f}" if not mdf.empty else "-",
            delta=None
        )

        # Metrics history plot
        st.subheader("History")
        if not mdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mdf["timestamp"],
                y=mdf["metric_value"],
                mode="lines+markers",
                name=METRIC_LABELS.get(metric, metric)
            ))
            fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title=METRIC_LABELS.get(metric, metric),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for this metric.")

        # Scenario breakdown
        st.subheader("Scenario Breakdown (latest run)")
        latest_run = sdf["run_id"].iloc[-1] if not sdf.empty else None
        if latest_run:
            ldf = df[(df["run_id"] == latest_run) & (df["scenario"] == scenario)]
            # Remove duplicates - keep only unique metric names
            ldf = ldf.drop_duplicates(subset=["metric_name"], keep="last")
            st.dataframe(ldf[["metric_name", "metric_value", "threshold_min", "threshold_max", "status"]])
        else:
            st.info("No scenario data available.")

        # Alerts with detailed explanations (expandable)
        st.subheader("⚠️ Alerts & Issues")
        alerts = ldf[ldf["status"] != "healthy"] if latest_run else pd.DataFrame()
        if not alerts.empty:
            for idx, row in alerts.iterrows():
                metric = row['metric_name']
                value = row['metric_value']
                status = row['status']
                thresh_min = row.get('threshold_min')
                thresh_max = row.get('threshold_max')

                # Generate detailed explanations for each metric issue
                alert_info = get_alert_explanation(metric, value, status, thresh_min, thresh_max)

                # Expandable alert - shows title, expands to show details
                with st.expander(alert_info['title'], expanded=False):
                    st.markdown(f"**What happened:**\n\n{alert_info['what']}")
                    st.markdown(f"**Why it matters:**\n\n{alert_info['why']}")
                    st.markdown(f"**How to fix:**\n\n{alert_info['fix']}")
        else:
            st.success("✅ All metrics healthy - system is performing within expected thresholds.")

        # --- Test Set and Case Breakdown ---
        st.header(f"🧪 Test Set & Case Results for {scenario}")

        # Load test case definitions for detail view
        test_case_defs = load_test_case_definitions()

        # Map scenarios to their prompt versions (from settings.yaml)
        SCENARIO_PROMPTS = {
            "hallucination_detection": "v1_zero_shot",
            "factual_accuracy": "v2_few_shot",
            "reasoning_quality": "v3_chain_of_thought",
            "instruction_following": "v4_rubric_based",
            "safety_compliance": "v5_structured_output",
            "consistency": "v1_zero_shot",
        }

        # Try to load detailed test case results for the selected scenario/run
        result_files = sorted(glob.glob("data/results/*.json") + glob.glob("data/daily_runs/*.json"), reverse=True)
        test_cases = []
        for f in result_files:
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                    # If file is a list of dicts (test cases)
                    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                        test_cases.extend(data)
                    # If file is a dict with test cases
                    elif isinstance(data, dict) and "test_cases" in data:
                        test_cases.extend(data["test_cases"])
            except Exception:
                pass

        if test_cases:
            df_cases = pd.DataFrame(test_cases)

            # Filter by scenario - match prompt_id or scenario field
            expected_prompt = SCENARIO_PROMPTS.get(scenario)
            if "scenario" in df_cases.columns:
                df_cases = df_cases[df_cases["scenario"] == scenario]
            elif "prompt_id" in df_cases.columns and expected_prompt:
                df_cases = df_cases[df_cases["prompt_id"] == expected_prompt]

            # Create results lookup by test_case_id
            filtered_cases = df_cases.to_dict('records')
            results_by_id = {tc.get("test_case_id"): tc for tc in filtered_cases if tc.get("test_case_id")}

            # Check if we have results for this scenario
            if df_cases.empty:
                st.info(f"No test case results found for scenario: {scenario}")
            else:
                # Interactive test case selector
                st.subheader("📋 Click a Test Case to View Details")

                # Get unique test case IDs
                if "test_case_id" in df_cases.columns:
                    case_ids = df_cases["test_case_id"].unique().tolist()

                    # Create a selectbox for test case selection (key includes scenario for proper reset)
                    selected_case = st.selectbox(
                        "Select Test Case",
                        options=["-- Select a test case --"] + case_ids,
                        key=f"test_case_selector_{scenario}"
                    )

                    # Show detail view if a case is selected
                    if selected_case and selected_case != "-- Select a test case --":
                        with st.container():
                            st.markdown("---")
                            result_data = results_by_id.get(selected_case, {})
                            render_test_case_detail(selected_case, test_case_defs, result_data)
                            st.markdown("---")

                # Show summary table with clickable-style formatting
                st.subheader(f"📊 Test Results for {scenario}")

                # Add status icons to the dataframe for visual clarity
                if "correct" in df_cases.columns:
                    df_cases["status"] = df_cases["correct"].apply(lambda x: "✅" if x else "❌")

                columns = [c for c in ["test_case_id", "status", "ground_truth", "prediction", "confidence", "prompt_id"] if c in df_cases.columns]
                st.dataframe(
                    df_cases[columns],
                    use_container_width=True,
                    column_config={
                        "test_case_id": st.column_config.TextColumn("Test Case ID", help="Select from dropdown above to view details"),
                        "status": st.column_config.TextColumn("Result", width="small"),
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    }
                )
                st.caption("💡 Tip: Use the dropdown above to view full details of any test case.")

                # === TEST RESULTS SUMMARY & NEXT STEPS ===
                render_test_summary(df_cases)
        else:
            st.info("No detailed test case results found for this scenario/run.")
    except Exception as e:
        st.exception(e)


def render_test_summary(df_cases: pd.DataFrame):
    """Render test results summary with next steps and investigation guidance."""

    st.header("📋 Test Results Summary & Next Steps")

    # --- Overall Summary ---
    total = len(df_cases)
    if "correct" in df_cases.columns:
        passed = df_cases["correct"].sum()
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
    else:
        passed = failed = 0
        pass_rate = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", total)
    with col2:
        st.metric("Passed", int(passed), delta=None)
    with col3:
        st.metric("Failed", int(failed), delta=None, delta_color="inverse")
    with col4:
        color = "🟢" if pass_rate >= 75 else "🟡" if pass_rate >= 60 else "🔴"
        st.metric("Pass Rate", f"{pass_rate:.1f}% {color}")

    # --- Failure Breakdown by Type ---
    st.subheader("🔍 Failure Analysis")

    if "correct" in df_cases.columns and failed > 0:
        failed_cases = df_cases[df_cases["correct"] == False].copy()

        # Extract failure mode from test_case_id (e.g., FM1_001 -> FM1)
        if "test_case_id" in failed_cases.columns:
            failed_cases["failure_mode"] = failed_cases["test_case_id"].str.extract(r"(FM\d+|REG)", expand=False).fillna("Unknown")

            failure_counts = failed_cases["failure_mode"].value_counts()

            # Failure mode descriptions
            FM_DESCRIPTIONS = {
                "FM1": "Factual Addition - True facts not in context",
                "FM2": "Fabrication - Outright false claims",
                "FM3": "Subtle Distortion - Small changes to facts",
                "FM4": "Valid Inference - Logical inferences incorrectly flagged",
                "FM5": "Verbatim Grounded - Direct quotes incorrectly flagged",
                "FM6": "Fluent Hallucination - Well-written but wrong",
                "FM7": "Partial Grounding - Mixed grounded/hallucinated",
                "REG": "Regression Test Cases",
                "Unknown": "Uncategorized failures"
            }

            st.write("**Failures by Category:**")
            for fm, count in failure_counts.items():
                desc = FM_DESCRIPTIONS.get(fm, "Unknown category")
                pct = count / failed * 100
                st.write(f"- **{fm}** ({count} failures, {pct:.0f}%): {desc}")

            # Show failed cases table
            with st.expander("📄 View Failed Test Cases", expanded=False):
                fail_cols = [c for c in ["test_case_id", "ground_truth", "prediction", "confidence", "llm_output"] if c in failed_cases.columns]
                st.dataframe(failed_cases[fail_cols])
    else:
        st.success("No failures detected!")

    # --- Next Steps Based on Results ---
    st.subheader("📝 Recommended Next Steps")

    if pass_rate >= 75:
        st.success("**Status: PRODUCTION READY**")
        st.markdown("""
        1. ✅ Run regression tests on held-out set (`REG_*` cases)
        2. ✅ Set up drift monitoring baseline
        3. ✅ Deploy with confidence monitoring enabled
        """)
    elif pass_rate >= 60:
        st.warning("**Status: NEEDS IMPROVEMENT**")
        st.markdown("""
        1. 🔄 Review failed cases in the table above
        2. 🔄 Consider upgrading prompt strategy:
           - Current < v3? Try **Chain-of-Thought** (v3)
           - Current v3? Try **Rubric-based** (v4) or **Structured Output** (v5)
        3. 🔄 Check if failures cluster in specific failure modes
        4. 🔄 Add more few-shot examples for problem categories
        """)
    else:
        st.error("**Status: SIGNIFICANT ISSUES**")
        st.markdown("""
        1. ❌ Do NOT deploy - accuracy too low
        2. ❌ Analyze failure patterns in detail (see below)
        3. ❌ Consider fundamental prompt redesign
        4. ❌ Review test case labels for correctness
        """)

    # --- Investigation Guide ---
    st.subheader("🔬 How to Investigate Failures")

    with st.expander("Investigation Checklist", expanded=True):
        st.markdown("""
        **Step 1: Identify the Pattern**
        - Are failures concentrated in one failure mode (FM1-FM7)?
        - Are failures mostly False Positives (flagging grounded as hallucination)?
        - Or False Negatives (missing actual hallucinations)?

        **Step 2: Examine Specific Cases**
        - Click "View Failed Test Cases" above
        - Look at `llm_output` to see model's reasoning
        - Compare `ground_truth` vs `prediction`

        **Step 3: Check Confidence Scores**
        - Low confidence on failures? Model is uncertain - add examples
        - High confidence on failures? Model has wrong understanding - revise prompt

        **Step 4: Take Action**
        | Issue | Solution |
        |-------|----------|
        | High FM1 failures | Add examples of factual additions to prompt |
        | High FM3 failures | Emphasize checking for subtle distortions |
        | High FM6 failures | Add "fluent ≠ correct" guidance |
        | Many False Positives | Increase TNR focus, relax criteria |
        | Many False Negatives | Increase recall focus, tighten criteria |

        **Step 5: Re-run & Compare**
        ```bash
        # Run evaluation with new prompt
        python -c "from src.evaluator import Evaluator; e = Evaluator(); print(e.evaluate_batch('v3_chain_of_thought'))"

        # Compare with A/B testing
        python -c "from src.ab_testing import ABTester; t = ABTester(); t.run_test(results_a, results_b, 'v2', 'v3')"
        ```
        """)

    # --- Quick Actions ---
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        st.code("# Re-run evaluation\npython -m src.orchestrator", language="bash")
    with col2:
        st.code("# Run A/B test\npython -c \"from src.ab_testing import ABTester; ABTester().run_test(...)\"", language="bash")


if __name__ == "__main__":
    main()
