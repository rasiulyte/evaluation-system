"""
LLM Evaluation Dashboard - rasar.ai
A thoughtful, minimal dashboard for AI evaluation metrics.

Design Philosophy: Intellectual, understated, quietly confident.
Technical depth without pretension. Quality over flash.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

# Import database abstraction
import sys
sys.path.insert(0, "src")
from database import db


# ============================================
# BRAND COLORS - rasar.ai palette
# ============================================

COLORS = {
    # Primary
    "navy": "#2c3e50",           # Main text, headers
    "charcoal": "#4a4a4a",       # Body text
    "white": "#f8f9fa",          # Background

    # Accent
    "teal": "#5a9a9c",           # Links, highlights
    "amber": "#c89f6f",          # CTAs, attention

    # Neutrals
    "light_gray": "#e8e8e8",     # Borders, dividers
    "medium_gray": "#8c8c8c",    # Secondary text

    # Status - muted red/green for clear pass/fail
    "good": "#4a8c6f",           # Muted forest green - passing/healthy
    "warning": "#c89f6f",        # Amber - needs attention
    "poor": "#b54a4a",           # Muted red - failing/critical
}


# ============================================
# METRIC INTERPRETATIONS - Beginner-friendly feedback
# ============================================

METRIC_INFO = {
    "f1": {
        "name": "F1 Score",
        "mental_model": "Like a smoke detector: catches real fires without false alarms when cooking",
        "scale": [
            (0.95, 1.00, "Exceptional", "Ready for high-stakes production"),
            (0.85, 0.95, "Excellent", "Production ready"),
            (0.75, 0.85, "Good", "Acceptable, minor tuning helpful"),
            (0.65, 0.75, "Fair", "Needs improvement before production"),
            (0.50, 0.65, "Poor", "Significant changes needed"),
            (0.00, 0.50, "Failing", "Fundamental redesign required"),
        ],
        "target": ">= 0.75",
    },
    "precision": {
        "name": "Precision",
        "mental_model": "Like a doctor's diagnosis: when they say 'you have X', how often are they right?",
        "scale": [
            (0.95, 1.00, "Exceptional", "Users fully trust warnings"),
            (0.90, 0.95, "Excellent", "Rare false alarms"),
            (0.80, 0.90, "Good", "Occasional false alarms, acceptable"),
            (0.70, 0.80, "Fair", "Noticeable false alarms"),
            (0.60, 0.70, "Poor", "Users start ignoring warnings"),
            (0.00, 0.60, "Failing", "Warnings are meaningless"),
        ],
        "target": ">= 0.75",
    },
    "recall": {
        "name": "Recall",
        "mental_model": "Like airport security: how many prohibited items do they actually catch?",
        "scale": [
            (0.95, 1.00, "Exceptional", "Almost nothing slips through"),
            (0.90, 0.95, "Excellent", "Very few missed"),
            (0.80, 0.90, "Good", "Some slip through, usually acceptable"),
            (0.70, 0.80, "Fair", "Notable blind spots"),
            (0.60, 0.70, "Poor", "Many missed"),
            (0.00, 0.60, "Failing", "More missed than caught"),
        ],
        "target": ">= 0.75",
    },
    "tnr": {
        "name": "TNR (Specificity)",
        "mental_model": "Like email spam filter: do important emails rarely go to spam?",
        "scale": [
            (0.95, 1.00, "Exceptional", "Good content flows freely"),
            (0.85, 0.95, "Excellent", "Rare blocking of good content"),
            (0.75, 0.85, "Good", "Occasional false blocks"),
            (0.65, 0.75, "Fair", "Users notice false blocks"),
            (0.55, 0.65, "Poor", "Frustrating user experience"),
            (0.00, 0.55, "Failing", "Blocks more good than bad"),
        ],
        "target": ">= 0.65",
    },
    "accuracy": {
        "name": "Accuracy",
        "mental_model": "Like weather forecast: what % of predictions are correct?",
        "scale": [
            (0.95, 1.00, "Exceptional", "Highly reliable"),
            (0.85, 0.95, "Excellent", "Very reliable"),
            (0.75, 0.85, "Good", "Generally reliable"),
            (0.65, 0.75, "Fair", "Sometimes unreliable"),
            (0.50, 0.65, "Poor", "Often wrong"),
            (0.00, 0.50, "Failing", "Worse than random guessing"),
        ],
        "target": ">= 0.75",
    },
    "cohens_kappa": {
        "name": "Cohen's Kappa",
        "mental_model": "Like two doctors diagnosing same patients: how much do they agree beyond luck?",
        "scale": [
            (0.81, 1.00, "Almost Perfect", "Extremely reliable"),
            (0.61, 0.81, "Substantial", "Good reliability"),
            (0.41, 0.61, "Moderate", "Fair reliability"),
            (0.21, 0.41, "Fair", "Limited reliability"),
            (0.00, 0.21, "Slight", "Poor reliability"),
            (-1.0, 0.00, "Poor", "Worse than chance"),
        ],
        "target": ">= 0.60",
    },
    "spearman": {
        "name": "Spearman Correlation",
        "mental_model": "Like student self-assessment: when confident, are they actually right?",
        "scale": [
            (0.80, 1.00, "Strong", "Confidence is very trustworthy"),
            (0.60, 0.80, "Moderate", "Confidence is useful but imperfect"),
            (0.40, 0.60, "Weak", "Confidence only somewhat helpful"),
            (0.20, 0.40, "Very Weak", "Confidence barely meaningful"),
            (0.00, 0.20, "None", "Confidence is random noise"),
            (-1.0, 0.00, "Negative", "Confidence is backwards!"),
        ],
        "target": ">= 0.60",
    },
    "pearson": {
        "name": "Pearson Correlation",
        "mental_model": "Is there a straight-line relationship between confidence and correctness?",
        "scale": [
            (0.80, 1.00, "Strong", "Linear relationship"),
            (0.60, 0.80, "Moderate", "Moderate relationship"),
            (0.40, 0.60, "Weak", "Weak relationship"),
            (0.20, 0.40, "Very Weak", "Minimal relationship"),
            (0.00, 0.20, "None", "No relationship"),
            (-1.0, 0.00, "Negative", "Inverse relationship"),
        ],
        "target": ">= 0.60",
    },
    "kendalls_tau": {
        "name": "Kendall's Tau",
        "mental_model": "Like ranking horses: do your predictions match actual race results?",
        "scale": [
            (0.70, 1.00, "Strong", "Excellent ranking alignment"),
            (0.50, 0.70, "Moderate", "Good ranking alignment"),
            (0.30, 0.50, "Weak", "Some ranking alignment"),
            (0.10, 0.30, "Very Weak", "Minimal alignment"),
            (0.00, 0.10, "None", "Rankings are unrelated"),
            (-1.0, 0.00, "Negative", "Rankings are reversed"),
        ],
        "target": ">= 0.50",
    },
    "bias": {
        "name": "Bias",
        "mental_model": "Like a scale not zeroed: does it consistently read heavy or light?",
        "scale": [
            (0.20, 1.00, "Strong Positive", "Way too aggressive - flags too much"),
            (0.10, 0.20, "Moderate Positive", "Somewhat aggressive"),
            (-0.10, 0.10, "Balanced", "Fair and unbiased"),
            (-0.20, -0.10, "Moderate Negative", "Somewhat lenient"),
            (-1.0, -0.20, "Strong Negative", "Way too lenient - misses too much"),
        ],
        "target": "|bias| <= 0.15",
        "lower_is_better": True,
        "absolute": True,
    },
    "mae": {
        "name": "MAE",
        "mental_model": "Like GPS accuracy: on average, how far off are your confidence scores?",
        "scale": [
            (0.00, 0.10, "Excellent", "Near-perfect calibration"),
            (0.10, 0.20, "Good", "Well calibrated"),
            (0.20, 0.30, "Fair", "Acceptable calibration"),
            (0.30, 0.40, "Poor", "Needs calibration work"),
            (0.40, 1.00, "Failing", "Severely miscalibrated"),
        ],
        "target": "< 0.20",
        "lower_is_better": True,
    },
    "rmse": {
        "name": "RMSE",
        "mental_model": "Like judging an archer: penalizes shots that completely miss the target",
        "scale": [
            (0.00, 0.15, "Excellent", "Small, consistent errors"),
            (0.15, 0.25, "Good", "Reasonable error distribution"),
            (0.25, 0.35, "Fair", "Some large errors present"),
            (0.35, 0.50, "Poor", "Frequent large errors"),
            (0.50, 1.00, "Failing", "Major errors common"),
        ],
        "target": "< 0.25",
        "lower_is_better": True,
    },
}


def get_metric_interpretation(metric_name: str, value: float) -> dict:
    """Get interpretation for a metric value."""
    info = METRIC_INFO.get(metric_name, {})
    if not info:
        return {"level": "Unknown", "description": "", "mental_model": ""}

    scale = info.get("scale", [])
    lower_is_better = info.get("lower_is_better", False)
    use_absolute = info.get("absolute", False)

    check_value = abs(value) if use_absolute else value

    for low, high, level, description in scale:
        if lower_is_better:
            if low <= check_value < high:
                return {
                    "level": level,
                    "description": description,
                    "mental_model": info.get("mental_model", ""),
                    "target": info.get("target", ""),
                }
        else:
            if low <= check_value <= high:
                return {
                    "level": level,
                    "description": description,
                    "mental_model": info.get("mental_model", ""),
                    "target": info.get("target", ""),
                }

    return {
        "level": "Unknown",
        "description": "",
        "mental_model": info.get("mental_model", ""),
        "target": info.get("target", ""),
    }


# ============================================
# CUSTOM CSS - Minimal, Clean, Professional
# ============================================

def apply_brand_css():
    """Apply rasar.ai brand styling - intellectual, understated, confident."""
    st.markdown(f"""
    <style>
    /* ===== GLOBAL RESET & BASE ===== */

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Base typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: {COLORS['charcoal']};
    }}

    /* Main container - generous white space */
    .main .block-container {{
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1200px;
        background-color: {COLORS['white']};
    }}

    /* ===== SIDEBAR ===== */

    [data-testid="stSidebar"] {{
        background-color: {COLORS['white']};
        border-right: 1px solid {COLORS['light_gray']};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
        color: {COLORS['charcoal']};
    }}

    /* ===== TYPOGRAPHY ===== */

    h1, h2, h3 {{
        color: {COLORS['navy']} !important;
        font-weight: 500;
        letter-spacing: -0.02em;
    }}

    h1 {{
        font-size: 1.75rem !important;
        margin-bottom: 0.5rem !important;
    }}

    h2 {{
        font-size: 1.25rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}

    h3 {{
        font-size: 1rem !important;
        color: {COLORS['charcoal']} !important;
    }}

    p, li {{
        color: {COLORS['charcoal']};
        line-height: 1.6;
    }}

    /* ===== METRIC CARDS ===== */

    .metric-card {{
        background: white;
        border: 1px solid {COLORS['light_gray']};
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s ease;
    }}

    .metric-card:hover {{
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
    }}

    .metric-label {{
        font-size: 0.8rem;
        color: {COLORS['medium_gray']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: 600;
        color: {COLORS['navy']};
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        margin-bottom: 0.5rem;
    }}

    .metric-interpretation {{
        font-size: 0.875rem;
        color: {COLORS['charcoal']};
        line-height: 1.5;
    }}

    .metric-context {{
        font-size: 0.8rem;
        color: {COLORS['medium_gray']};
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid {COLORS['light_gray']};
    }}

    /* Status indicators - subtle, not flashy */
    .status-good {{
        border-left: 3px solid {COLORS['good']};
    }}

    .status-warning {{
        border-left: 3px solid {COLORS['amber']};
    }}

    .status-poor {{
        border-left: 3px solid {COLORS['poor']};
    }}

    /* Status badge - minimal */
    .status-badge {{
        display: inline-block;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-weight: 500;
    }}

    .badge-good {{
        background: rgba(74, 140, 111, 0.12);
        color: {COLORS['good']};
    }}

    .badge-warning {{
        background: rgba(200, 159, 111, 0.15);
        color: {COLORS['amber']};
    }}

    .badge-poor {{
        background: rgba(181, 74, 74, 0.12);
        color: {COLORS['poor']};
    }}

    /* ===== SECTION HEADERS ===== */

    .section-header {{
        color: {COLORS['navy']};
        font-size: 1.1rem;
        font-weight: 500;
        margin: 2.5rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid {COLORS['light_gray']};
    }}

    .section-subtitle {{
        font-size: 0.875rem;
        color: {COLORS['medium_gray']};
        margin-top: -0.75rem;
        margin-bottom: 1.5rem;
    }}

    /* ===== PAGE HEADER ===== */

    .page-header {{
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid {COLORS['light_gray']};
    }}

    .page-title {{
        font-size: 1.5rem;
        font-weight: 500;
        color: {COLORS['navy']};
        margin: 0;
    }}

    .page-subtitle {{
        font-size: 0.9rem;
        color: {COLORS['medium_gray']};
        margin-top: 0.5rem;
    }}

    /* ===== BUTTONS ===== */

    .stButton > button {{
        background-color: {COLORS['navy']} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }}

    .stButton > button:hover {{
        background-color: #3d5266 !important;
        color: white !important;
    }}

    /* Primary button - teal for action */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background-color: {COLORS['teal']} !important;
        color: white !important;
        border: none !important;
    }}

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background-color: #4a8a8c !important;
        color: white !important;
    }}

    /* Secondary button style */
    .stButton > button[kind="secondary"] {{
        background-color: white !important;
        color: {COLORS['navy']} !important;
        border: 1px solid {COLORS['light_gray']} !important;
    }}

    /* ===== DATAFRAMES ===== */

    .stDataFrame {{
        border: 1px solid {COLORS['light_gray']};
        border-radius: 8px;
    }}

    /* ===== EXPANDERS ===== */

    .streamlit-expanderHeader {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {COLORS['charcoal']};
        background-color: transparent;
    }}

    /* ===== DIVIDERS ===== */

    hr {{
        border: none;
        border-top: 1px solid {COLORS['light_gray']};
        margin: 2rem 0;
    }}

    /* ===== ALERTS - Understated ===== */

    .stAlert {{
        border-radius: 6px;
        border-left-width: 3px;
    }}

    /* ===== TABS ===== */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        border-bottom: 1px solid {COLORS['light_gray']};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['medium_gray']};
        font-weight: 500;
        padding: 0.75rem 0;
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS['navy']};
        border-bottom-color: {COLORS['teal']};
    }}

    /* ===== SELECTBOX ===== */

    .stSelectbox label {{
        color: {COLORS['charcoal']};
        font-size: 0.875rem;
    }}

    /* ===== LINKS ===== */

    a {{
        color: {COLORS['teal']};
        text-decoration: none;
    }}

    a:hover {{
        text-decoration: underline;
    }}

    /* ===== FOOTER ===== */

    .footer {{
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid {COLORS['light_gray']};
        text-align: center;
        color: {COLORS['medium_gray']};
        font-size: 0.8rem;
    }}

    .footer a {{
        color: {COLORS['teal']};
    }}

    /* ===== RESPONSIVE ===== */

    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem 1.5rem;
        }}

        .metric-value {{
            font-size: 1.5rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================
# COMPONENT HELPERS
# ============================================

def render_page_header(title: str, subtitle: str = None):
    """Render a clean page header."""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = None):
    """Render a section header with optional subtitle."""
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="section-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def get_status_class(value: float, thresholds: dict) -> tuple:
    """
    Determine status based on value and thresholds.
    Returns (css_class, badge_class, status_text)
    """
    good_min = thresholds.get("good_min")
    warning_min = thresholds.get("warning_min")
    higher_is_better = thresholds.get("higher_is_better", True)

    if higher_is_better:
        if good_min and value >= good_min:
            return "status-good", "badge-good", "Good"
        elif warning_min and value >= warning_min:
            return "status-warning", "badge-warning", "Fair"
        else:
            return "status-poor", "badge-poor", "Needs Attention"
    else:
        # For metrics where lower is better (like bias, MAE)
        good_max = thresholds.get("good_max", 0.15)
        warning_max = thresholds.get("warning_max", 0.25)
        if value <= good_max:
            return "status-good", "badge-good", "Good"
        elif value <= warning_max:
            return "status-warning", "badge-warning", "Fair"
        else:
            return "status-poor", "badge-poor", "Needs Attention"


def render_metric_card(
    name: str,
    value: float,
    interpretation: str,
    context: str,
    thresholds: dict,
    format_str: str = ".3f",
    metric_key: str = None
):
    """
    Render a metric card with consistent styling and interpretation feedback.

    Args:
        name: Metric display name
        value: Numeric value
        interpretation: What this value means
        context: When this metric matters
        thresholds: Dict with good_min, warning_min, higher_is_better
        format_str: Format string for value display
        metric_key: Key to look up in METRIC_INFO for detailed interpretation
    """
    status_class, badge_class, status_text = get_status_class(value, thresholds)

    formatted_value = f"{value:{format_str}}" if isinstance(value, (int, float)) and not (value != value) else "N/A"

    # Get detailed interpretation if metric_key provided
    interp_html = ""
    if metric_key and metric_key in METRIC_INFO:
        interp = get_metric_interpretation(metric_key, value)
        if interp["level"] != "Unknown":
            level_color = COLORS["good"] if status_class == "status-good" else (
                COLORS["poor"] if status_class == "status-poor" else COLORS["amber"]
            )
            interp_html = f"""
            <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid {COLORS['light_gray']};">
                <div style="font-size: 0.75rem; color: {level_color}; font-weight: 500; margin-bottom: 0.25rem;">
                    {interp["level"]}: {interp["description"]}
                </div>
                <div style="font-size: 0.7rem; color: {COLORS['medium_gray']}; font-style: italic;">
                    {interp["mental_model"]}
                </div>
            </div>
            """

    st.markdown(f"""
    <div class="metric-card {status_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <span class="metric-label">{name}</span>
            <span class="status-badge {badge_class}">{status_text}</span>
        </div>
        <div class="metric-value">{formatted_value}</div>
        <div class="metric-interpretation">{interpretation}</div>
        <div class="metric-context">{context}</div>
        {interp_html}
    </div>
    """, unsafe_allow_html=True)


def render_simple_metric(name: str, value: str, sublabel: str = None):
    """Render a simple metric without status indicator."""
    sublabel_html = f'<div style="font-size: 0.8rem; color: {COLORS["medium_gray"]};">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">{name}</span>
        <div class="metric-value">{value}</div>
        {sublabel_html}
    </div>
    """, unsafe_allow_html=True)


def apply_chart_theme(fig):
    """Apply rasar.ai brand theme to Plotly charts - clean and minimal."""
    fig.update_layout(
        font=dict(
            family="Inter, -apple-system, sans-serif",
            color=COLORS["charcoal"],
            size=12
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=40, b=40, l=50, r=30),

        # Subtle grid
        xaxis=dict(
            gridcolor=COLORS["light_gray"],
            linecolor=COLORS["light_gray"],
            tickcolor=COLORS["medium_gray"],
            title_font=dict(color=COLORS["charcoal"], size=11),
            tickfont=dict(color=COLORS["medium_gray"], size=10)
        ),
        yaxis=dict(
            gridcolor=COLORS["light_gray"],
            linecolor=COLORS["light_gray"],
            tickcolor=COLORS["medium_gray"],
            title_font=dict(color=COLORS["charcoal"], size=11),
            tickfont=dict(color=COLORS["medium_gray"], size=10)
        ),

        # Clean legend
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["light_gray"],
            borderwidth=1,
            font=dict(size=11)
        ),

        # Use brand colors
        colorway=[COLORS["teal"], COLORS["navy"], COLORS["amber"], COLORS["medium_gray"]]
    )
    return fig


def render_footer():
    """Render the footer."""
    st.markdown(f"""
    <div class="footer">
        <p>A sandbox for learning AI evaluation</p>
        <p><a href="https://rasar.ai" target="_blank">rasar.ai</a></p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# DATA LOADING
# ============================================

def load_metrics() -> pd.DataFrame:
    """Load metrics from database."""
    try:
        metrics = db.get_all_metrics()
        if metrics:
            return pd.DataFrame(metrics)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_test_results(scenario: str = None, run_id: str = None) -> pd.DataFrame:
    """Load test results from database."""
    try:
        results = db.get_test_results(run_id=run_id, scenario=scenario)
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_latest_metrics(df: pd.DataFrame, scenario: str = None) -> dict:
    """Get the most recent metrics as a dictionary."""
    if df.empty:
        return {}

    if scenario:
        df = df[df["scenario"] == scenario]

    if df.empty:
        return {}

    # Get latest run
    latest_run = df.sort_values("timestamp", ascending=False)["run_id"].iloc[0]
    latest_df = df[df["run_id"] == latest_run]

    metrics = {}
    for _, row in latest_df.iterrows():
        metrics[row["metric_name"]] = {
            "value": row["metric_value"],
            "status": row.get("status", "unknown"),
            "threshold_min": row.get("threshold_min"),
            "threshold_max": row.get("threshold_max")
        }

    return metrics


# ============================================
# METRIC DEFINITIONS
# ============================================

METRIC_INFO = {
    # Correlation Metrics
    "kendalls_tau": {
        "name": "Kendall's Tau",
        "category": "correlation",
        "thresholds": {"good_min": 0.6, "warning_min": 0.4, "higher_is_better": True},
        "interpretation": lambda v: (
            "Strong agreement between rankings" if v >= 0.6 else
            "Moderate agreement" if v >= 0.4 else
            "Weak agreement - model confidence poorly predicts correctness"
        ),
        "context": "Measures rank correlation between confidence scores and actual correctness. More robust to outliers than Pearson."
    },
    "pearson": {
        "name": "Pearson Correlation",
        "category": "correlation",
        "thresholds": {"good_min": 0.7, "warning_min": 0.5, "higher_is_better": True},
        "interpretation": lambda v: (
            "Strong linear relationship" if v >= 0.7 else
            "Moderate linear relationship" if v >= 0.5 else
            "Weak linear relationship - confidence scores not well calibrated"
        ),
        "context": "Measures linear correlation between confidence and correctness. Use when you expect a proportional relationship."
    },
    "spearman": {
        "name": "Spearman Correlation",
        "category": "correlation",
        "thresholds": {"good_min": 0.7, "warning_min": 0.5, "higher_is_better": True},
        "interpretation": lambda v: (
            "Strong monotonic relationship" if v >= 0.7 else
            "Moderate monotonic relationship" if v >= 0.5 else
            "Weak relationship"
        ),
        "context": "Rank-based correlation. Less sensitive to outliers than Pearson."
    },

    # Agreement Metrics
    "cohens_kappa": {
        "name": "Cohen's Kappa",
        "category": "agreement",
        "thresholds": {"good_min": 0.61, "warning_min": 0.41, "higher_is_better": True},
        "interpretation": lambda v: (
            "Substantial agreement" if v >= 0.61 else
            "Moderate agreement" if v >= 0.41 else
            "Fair agreement" if v >= 0.21 else
            "Slight agreement - little better than chance"
        ),
        "context": "Agreement corrected for chance. Standard interpretation: 0.81-1.0 almost perfect, 0.61-0.80 substantial, 0.41-0.60 moderate."
    },

    # Classification Metrics
    "precision": {
        "name": "Precision",
        "category": "classification",
        "thresholds": {"good_min": 0.75, "warning_min": 0.60, "higher_is_better": True},
        "interpretation": lambda v: (
            f"When flagging hallucinations, {v*100:.0f}% are correct" if v > 0 else
            "No precision data"
        ),
        "context": "Of all predicted hallucinations, what fraction were actual hallucinations? High precision = trustworthy alerts."
    },
    "recall": {
        "name": "Recall",
        "category": "classification",
        "thresholds": {"good_min": 0.75, "warning_min": 0.60, "higher_is_better": True},
        "interpretation": lambda v: (
            f"Catching {v*100:.0f}% of actual hallucinations" if v > 0 else
            "No recall data"
        ),
        "context": "Of all actual hallucinations, what fraction were detected? High recall = few missed hallucinations."
    },
    "f1": {
        "name": "F1 Score",
        "category": "classification",
        "thresholds": {"good_min": 0.75, "warning_min": 0.60, "higher_is_better": True},
        "interpretation": lambda v: (
            "Excellent balance of precision and recall" if v >= 0.75 else
            "Good balance" if v >= 0.60 else
            "Needs improvement - check precision vs recall individually"
        ),
        "context": "Harmonic mean of precision and recall. Primary metric for overall detection quality."
    },
    "accuracy": {
        "name": "Accuracy",
        "category": "classification",
        "thresholds": {"good_min": 0.80, "warning_min": 0.65, "higher_is_better": True},
        "interpretation": lambda v: (
            f"{v*100:.0f}% of all predictions are correct" if v > 0 else
            "No accuracy data"
        ),
        "context": "Overall correctness. Can be misleading with imbalanced classes - use F1 for better insight."
    },
    "tnr": {
        "name": "True Negative Rate",
        "category": "classification",
        "thresholds": {"good_min": 0.70, "warning_min": 0.55, "higher_is_better": True},
        "interpretation": lambda v: (
            f"Correctly accepting {v*100:.0f}% of grounded content" if v > 0 else
            "No TNR data"
        ),
        "context": "Of all grounded content, what fraction was correctly accepted? High TNR = few false alarms."
    },

    # Error/Bias Metrics
    "bias": {
        "name": "Prediction Bias",
        "category": "error",
        "thresholds": {"good_max": 0.10, "warning_max": 0.20, "higher_is_better": False},
        "interpretation": lambda v: (
            "Balanced predictions" if abs(v) <= 0.10 else
            f"{'Over-predicting' if v > 0 else 'Under-predicting'} hallucinations by {abs(v)*100:.0f}%"
        ),
        "context": "Systematic tendency to over/under-predict. Positive = too many false positives. Negative = too many false negatives."
    },
    "mae": {
        "name": "Mean Absolute Error",
        "category": "error",
        "thresholds": {"good_max": 0.15, "warning_max": 0.25, "higher_is_better": False},
        "interpretation": lambda v: (
            "Well-calibrated confidence" if v <= 0.15 else
            "Moderate calibration error" if v <= 0.25 else
            "Poor calibration - confidence scores don't reflect true accuracy"
        ),
        "context": "Average magnitude of confidence calibration errors. Lower is better."
    },
    "rmse": {
        "name": "Root Mean Square Error",
        "category": "error",
        "thresholds": {"good_max": 0.20, "warning_max": 0.30, "higher_is_better": False},
        "interpretation": lambda v: (
            "Low error variance" if v <= 0.20 else
            "Moderate error variance" if v <= 0.30 else
            "High error variance - some predictions are very wrong"
        ),
        "context": "Like MAE but penalizes large errors more heavily. Sensitive to outliers."
    }
}


# ============================================
# PAGE: METRICS OVERVIEW
# ============================================

def render_metrics_overview_page(df: pd.DataFrame):
    """Main metrics dashboard - organized by category."""

    render_page_header(
        "Evaluation Metrics",
        "Current performance across all metric categories"
    )

    if df.empty:
        st.info("No evaluation data yet. Run an evaluation to see metrics here.")
        return

    # Get scenario selector
    scenarios = sorted(df["scenario"].unique())
    selected_scenario = st.selectbox(
        "Select Scenario",
        scenarios,
        key="overview_scenario"
    )

    # Get latest metrics for this scenario
    metrics = get_latest_metrics(df, selected_scenario)

    if not metrics:
        st.info(f"No metrics found for scenario: {selected_scenario}")
        return

    # Get run info
    scenario_df = df[df["scenario"] == selected_scenario]
    latest_run = scenario_df.sort_values("timestamp", ascending=False).iloc[0]

    st.caption(f"Latest run: {latest_run['run_id']} · {latest_run['timestamp'][:16]}")

    # Show available metrics (debug info)
    available_metrics = list(metrics.keys())
    expected_metrics = ["f1", "precision", "recall", "tnr", "accuracy", "cohens_kappa", "bias", "spearman", "pearson", "kendalls_tau", "mae", "rmse"]
    missing_metrics = [m for m in expected_metrics if m not in available_metrics]

    if missing_metrics:
        with st.expander(f"ℹ️ {len(missing_metrics)} metrics not available", expanded=False):
            st.caption(f"**Available:** {', '.join(available_metrics)}")
            st.caption(f"**Missing:** {', '.join(missing_metrics)}")
            st.caption("Missing metrics may require running a new evaluation with the updated orchestrator.")

    # --- SECTION 1: Classification Metrics ---
    render_section_header(
        "Classification Performance",
        "How well the model identifies hallucinations vs grounded content"
    )

    classification_metrics = ["f1", "precision", "recall", "tnr", "accuracy"]
    cols = st.columns(3)

    col_idx = 0
    for metric_key in classification_metrics:
        if metric_key in metrics:
            info = METRIC_INFO.get(metric_key, {})
            value = metrics[metric_key]["value"]

            with cols[col_idx % 3]:
                render_metric_card(
                    name=info.get("name", metric_key),
                    value=value,
                    interpretation=info.get("interpretation", lambda v: "")(value),
                    context=info.get("context", ""),
                    thresholds=info.get("thresholds", {}),
                    format_str=".2f",
                    metric_key=metric_key
                )
            col_idx += 1

    # --- SECTION 2: Agreement Metrics ---
    if "cohens_kappa" in metrics:
        render_section_header(
            "Agreement",
            "Consistency and reliability of predictions"
        )

        info = METRIC_INFO["cohens_kappa"]
        value = metrics["cohens_kappa"]["value"]

        col1, col2 = st.columns([1, 2])
        with col1:
            render_metric_card(
                name=info["name"],
                value=value,
                interpretation=info["interpretation"](value),
                context=info["context"],
                thresholds=info["thresholds"],
                format_str=".3f",
                metric_key="cohens_kappa"
            )

        with col2:
            # Kappa interpretation scale
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Kappa Interpretation Scale</span>
                <div style="margin-top: 1rem;">
                    <div style="display: flex; margin-bottom: 0.5rem;">
                        <span style="width: 120px; color: {COLORS['medium_gray']};">0.81 - 1.00</span>
                        <span style="color: {COLORS['charcoal']};">Almost Perfect</span>
                    </div>
                    <div style="display: flex; margin-bottom: 0.5rem;">
                        <span style="width: 120px; color: {COLORS['medium_gray']};">0.61 - 0.80</span>
                        <span style="color: {COLORS['charcoal']};">Substantial</span>
                    </div>
                    <div style="display: flex; margin-bottom: 0.5rem;">
                        <span style="width: 120px; color: {COLORS['medium_gray']};">0.41 - 0.60</span>
                        <span style="color: {COLORS['charcoal']};">Moderate</span>
                    </div>
                    <div style="display: flex; margin-bottom: 0.5rem;">
                        <span style="width: 120px; color: {COLORS['medium_gray']};">0.21 - 0.40</span>
                        <span style="color: {COLORS['charcoal']};">Fair</span>
                    </div>
                    <div style="display: flex;">
                        <span style="width: 120px; color: {COLORS['medium_gray']};">0.00 - 0.20</span>
                        <span style="color: {COLORS['charcoal']};">Slight</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- SECTION 3: Correlation Metrics ---
    import math
    correlation_metrics = [m for m in ["kendalls_tau", "spearman", "pearson"] if m in metrics]

    if correlation_metrics:
        render_section_header(
            "Correlation",
            "How well confidence scores predict actual correctness"
        )

        # Check if correlation values are valid (not NaN)
        valid_correlation_metrics = []
        for m in correlation_metrics:
            val = metrics[m]["value"]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                valid_correlation_metrics.append(m)

        if valid_correlation_metrics:
            cols = st.columns(len(valid_correlation_metrics))
            for i, metric_key in enumerate(valid_correlation_metrics):
                info = METRIC_INFO.get(metric_key, {})
                value = metrics[metric_key]["value"]

                with cols[i]:
                    render_metric_card(
                        name=info.get("name", metric_key),
                        value=value,
                        interpretation=info.get("interpretation", lambda v: "")(value),
                        context=info.get("context", ""),
                        thresholds=info.get("thresholds", {}),
                        format_str=".3f",
                        metric_key=metric_key
                    )
        else:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem;">
                <span class="metric-label">Correlation Metrics Unavailable</span>
                <div style="margin-top: 0.5rem; color: {COLORS['charcoal']};">
                    Correlation metrics require variable confidence scores from the LLM.
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: {COLORS['medium_gray']};">
                    <strong>To enable:</strong> Use the <code>v5_structured_output</code> prompt, which asks the LLM
                    to return confidence scores in JSON format.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- SECTION 4: Error Metrics ---
    error_metrics = [m for m in ["bias", "mae", "rmse"] if m in metrics]

    if error_metrics:
        render_section_header(
            "Error Analysis",
            "Systematic biases and calibration quality"
        )

        # Check for valid values
        valid_error_metrics = []
        for m in error_metrics:
            val = metrics[m]["value"]
            # bias is always valid, mae/rmse might be NaN or 0.5 (default)
            if m == "bias":
                valid_error_metrics.append(m)
            elif val is not None and not (isinstance(val, float) and math.isnan(val)):
                valid_error_metrics.append(m)

        if valid_error_metrics:
            cols = st.columns(len(valid_error_metrics))
            for i, metric_key in enumerate(valid_error_metrics):
                info = METRIC_INFO.get(metric_key, {})
                value = metrics[metric_key]["value"]

                with cols[i]:
                    render_metric_card(
                        name=info.get("name", metric_key),
                        value=value,
                        interpretation=info.get("interpretation", lambda v: "")(value),
                        context=info.get("context", ""),
                        thresholds=info.get("thresholds", {}),
                        format_str=".3f",
                        metric_key=metric_key
                    )


# ============================================
# PAGE: TRENDS
# ============================================

def render_trends_page(df: pd.DataFrame):
    """Historical trends visualization."""

    render_page_header(
        "Performance Trends",
        "Track metric changes over time"
    )

    if df.empty:
        st.info("No historical data available yet.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        scenarios = sorted(df["scenario"].unique())
        selected_scenario = st.selectbox("Scenario", scenarios, key="trend_scenario")

    with col2:
        available_metrics = df[df["scenario"] == selected_scenario]["metric_name"].unique()
        metric_options = [m for m in ["f1", "precision", "recall", "tnr", "cohens_kappa"] if m in available_metrics]
        if not metric_options:
            metric_options = list(available_metrics)
        selected_metric = st.selectbox("Metric", metric_options, key="trend_metric")

    # Filter data
    filtered = df[(df["scenario"] == selected_scenario) & (df["metric_name"] == selected_metric)]
    filtered = filtered.sort_values("timestamp")

    if filtered.empty:
        st.info("No data for this selection.")
        return

    # Create trend chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["metric_value"],
        mode="lines+markers",
        name=METRIC_INFO.get(selected_metric, {}).get("name", selected_metric),
        line=dict(color=COLORS["teal"], width=2),
        marker=dict(size=6, color=COLORS["navy"])
    ))

    # Add threshold line if available
    thresholds = METRIC_INFO.get(selected_metric, {}).get("thresholds", {})
    if thresholds.get("good_min"):
        fig.add_hline(
            y=thresholds["good_min"],
            line_dash="dash",
            line_color=COLORS["medium_gray"],
            annotation_text=f"Target: {thresholds['good_min']}",
            annotation_position="right"
        )

    fig.update_layout(
        title=f"{METRIC_INFO.get(selected_metric, {}).get('name', selected_metric)} Over Time",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )

    apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    render_section_header("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_simple_metric("Current", f"{filtered['metric_value'].iloc[-1]:.3f}")
    with col2:
        render_simple_metric("Average", f"{filtered['metric_value'].mean():.3f}")
    with col3:
        render_simple_metric("Min", f"{filtered['metric_value'].min():.3f}")
    with col4:
        render_simple_metric("Max", f"{filtered['metric_value'].max():.3f}")

    # Change analysis
    if len(filtered) >= 2:
        first_val = filtered["metric_value"].iloc[0]
        last_val = filtered["metric_value"].iloc[-1]
        change = last_val - first_val
        pct_change = (change / first_val * 100) if first_val != 0 else 0

        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Change from First to Latest</span>
            <div style="margin-top: 0.5rem; color: {COLORS['charcoal']};">
                {'+' if change >= 0 else ''}{change:.3f} ({'+' if pct_change >= 0 else ''}{pct_change:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# PAGE: RUN HISTORY
# ============================================

def render_run_history_page(df: pd.DataFrame):
    """List of all evaluation runs."""

    render_page_header(
        "Run History",
        "All evaluation runs and their outcomes"
    )

    if df.empty:
        st.info("No runs recorded yet.")
        return

    # Build runs summary
    runs_data = []
    for run_id in df["run_id"].unique():
        run_df = df[df["run_id"] == run_id]
        timestamp = run_df["timestamp"].iloc[0]

        # Get key metrics
        f1_val = run_df[run_df["metric_name"] == "f1"]["metric_value"].values
        f1 = f1_val[0] if len(f1_val) > 0 else None

        precision_val = run_df[run_df["metric_name"] == "precision"]["metric_value"].values
        precision = precision_val[0] if len(precision_val) > 0 else None

        recall_val = run_df[run_df["metric_name"] == "recall"]["metric_value"].values
        recall = recall_val[0] if len(recall_val) > 0 else None

        # Determine overall status
        if f1 and f1 >= 0.75:
            status = "✓ Passing"
            status_raw = "passing"
        elif f1 and f1 >= 0.60:
            status = "○ Fair"
            status_raw = "fair"
        else:
            status = "✗ Failing"
            status_raw = "failing"

        runs_data.append({
            "Run ID": run_id,
            "Date": timestamp[:10],
            "Time": timestamp[11:16] if len(timestamp) > 11 else "",
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Status": status,
            "status_raw": status_raw
        })

    runs_df = pd.DataFrame(runs_data).sort_values("Date", ascending=False)

    # Summary cards with red/green
    passing_runs = len(runs_df[runs_df["status_raw"] == "passing"])
    failing_runs = len(runs_df[runs_df["status_raw"] == "failing"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_simple_metric("Total Runs", str(len(runs_df)))
    with col2:
        st.markdown(f"""
        <div class="metric-card status-good">
            <span class="metric-label">Passing</span>
            <div class="metric-value" style="color: {COLORS['good']};">{passing_runs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card status-poor">
            <span class="metric-label">Failing</span>
            <div class="metric-value" style="color: {COLORS['poor']};">{failing_runs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        if not runs_df.empty:
            latest = runs_df.iloc[0]
            latest_status = latest["status_raw"]
            color = COLORS['good'] if latest_status == "passing" else (COLORS['poor'] if latest_status == "failing" else COLORS['amber'])
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Latest</span>
                <div class="metric-value" style="color: {color}; font-size: 1.25rem;">{latest["Status"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Display each run as a card with colored status
    for _, row in runs_df.iterrows():
        status_raw = row['status_raw']
        if status_raw == "passing":
            status_class = "status-good"
            status_color = COLORS['good']
            status_text = "✓ Passing"
        elif status_raw == "failing":
            status_class = "status-poor"
            status_color = COLORS['poor']
            status_text = "✗ Failing"
        else:
            status_class = "status-warning"
            status_color = COLORS['amber']
            status_text = "○ Fair"

        f1_str = f"{row['F1']:.3f}" if row['F1'] is not None else "—"
        prec_str = f"{row['Precision']:.3f}" if row['Precision'] is not None else "—"
        rec_str = f"{row['Recall']:.3f}" if row['Recall'] is not None else "—"

        st.markdown(f"""
        <div class="metric-card {status_class}" style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-family: monospace; font-size: 0.85rem; color: {COLORS['navy']};">{row['Run ID']}</span>
                <span style="color: {status_color}; font-weight: 500;">{status_text}</span>
            </div>
            <div style="display: flex; gap: 2rem; font-size: 0.85rem; color: {COLORS['medium_gray']};">
                <span>{row['Date']} {row['Time']}</span>
                <span>F1: <strong style="color: {COLORS['charcoal']};">{f1_str}</strong></span>
                <span>Precision: <strong style="color: {COLORS['charcoal']};">{prec_str}</strong></span>
                <span>Recall: <strong style="color: {COLORS['charcoal']};">{rec_str}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# PAGE: COMPARE RUNS
# ============================================

def get_runs_with_metrics(df: pd.DataFrame) -> list:
    """Get list of run_ids that have metrics data."""
    if df.empty:
        return []
    return df.sort_values("timestamp", ascending=False)["run_id"].unique().tolist()


def get_metrics_comparison(df: pd.DataFrame, run_id_1: str, run_id_2: str) -> pd.DataFrame:
    """Compare metrics between two runs."""
    if df.empty:
        return pd.DataFrame()

    m1 = df[df['run_id'] == run_id_1][['scenario', 'metric_name', 'metric_value']].copy()
    m2 = df[df['run_id'] == run_id_2][['scenario', 'metric_name', 'metric_value']].copy()

    merged = m1.merge(m2, on=['scenario', 'metric_name'], suffixes=('_baseline', '_compare'))
    merged['delta'] = merged['metric_value_compare'] - merged['metric_value_baseline']
    merged['delta_pct'] = merged.apply(
        lambda r: ((r['metric_value_compare'] - r['metric_value_baseline']) / r['metric_value_baseline'] * 100)
        if r['metric_value_baseline'] != 0 else 0,
        axis=1
    )
    return merged


def render_compare_runs_page(df: pd.DataFrame):
    """Compare metrics between two evaluation runs."""

    render_page_header(
        "Compare Runs",
        "Analyze performance differences between evaluation runs"
    )

    if df.empty:
        st.info("No evaluation data available yet.")
        return

    # Get available runs
    run_ids = get_runs_with_metrics(df)

    if len(run_ids) < 2:
        st.info(f"Need at least 2 runs to compare. Currently have {len(run_ids)} run(s).")
        return

    # Run selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Baseline Run**")
        baseline_run = st.selectbox(
            "Select baseline",
            run_ids,
            index=min(1, len(run_ids)-1),
            key="baseline_run",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown(f"**Compare Run**")
        compare_run = st.selectbox(
            "Select comparison",
            run_ids,
            index=0,
            key="compare_run",
            label_visibility="collapsed"
        )

    if baseline_run == compare_run:
        st.warning("Please select different runs to compare.")
        return

    # Get comparison data
    comparison_df = get_metrics_comparison(df, baseline_run, compare_run)

    if comparison_df.empty:
        st.warning("Could not generate comparison data.")
        return

    # Summary stats
    render_section_header("Summary")

    # Metrics where LOWER is better
    lower_is_better_metrics = {"bias", "mae", "rmse"}

    # Count improved/declined accounting for metric direction
    improved = 0
    declined = 0
    unchanged = 0

    for _, row in comparison_df.iterrows():
        delta = row["delta"]
        metric_name = row["metric_name"].lower()
        is_lower_better = metric_name in lower_is_better_metrics

        if abs(delta) <= 0.001:
            unchanged += 1
        elif delta > 0:
            # Value increased
            if is_lower_better:
                declined += 1  # Increase in bias = bad
            else:
                improved += 1  # Increase in f1 = good
        else:
            # Value decreased
            if is_lower_better:
                improved += 1  # Decrease in bias = good
            else:
                declined += 1  # Decrease in f1 = bad

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card status-good">
            <span class="metric-label">Improved</span>
            <div class="metric-value" style="color: {COLORS['good']};">{improved}</div>
            <div class="metric-interpretation">metrics got better</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card status-poor">
            <span class="metric-label">Declined</span>
            <div class="metric-value" style="color: {COLORS['poor']};">{declined}</div>
            <div class="metric-interpretation">metrics got worse</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Unchanged</span>
            <div class="metric-value">{unchanged}</div>
            <div class="metric-interpretation">metrics stable</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed comparison
    render_section_header("Detailed Comparison")

    # Scenario filter for detailed view
    scenarios = sorted(comparison_df["scenario"].unique())
    selected_detail_scenario = st.selectbox(
        "Filter by Scenario",
        ["All Scenarios"] + scenarios,
        key="detail_scenario"
    )

    # Filter data based on selection
    if selected_detail_scenario == "All Scenarios":
        filtered_comparison = comparison_df.copy()
    else:
        filtered_comparison = comparison_df[comparison_df["scenario"] == selected_detail_scenario].copy()

    # Define metric ordering (classification → agreement → correlation → error)
    metric_order = {
        "f1": 1, "precision": 2, "recall": 3, "tnr": 4, "accuracy": 5,
        "cohens_kappa": 6,
        "spearman": 7, "pearson": 8, "kendalls_tau": 9,
        "bias": 10, "mae": 11, "rmse": 12
    }
    filtered_comparison["_metric_order"] = filtered_comparison["metric_name"].map(
        lambda x: metric_order.get(x, 99)
    )
    filtered_comparison = filtered_comparison.sort_values(["scenario", "_metric_order"]).drop(columns=["_metric_order"])

    # Metrics where LOWER is better (increase = bad, decrease = good)
    lower_is_better = {"bias", "mae", "rmse"}

    # Define metric categories for grouping
    metric_categories = {
        "f1": "Classification", "precision": "Classification", "recall": "Classification",
        "tnr": "Classification", "accuracy": "Classification",
        "cohens_kappa": "Agreement",
        "spearman": "Correlation", "pearson": "Correlation", "kendalls_tau": "Correlation",
        "bias": "Calibration", "mae": "Calibration", "rmse": "Calibration"
    }

    # Track current category and scenario to show headers
    current_category = None
    current_scenario = None

    # Display comparison as styled rows with colored changes
    for _, row in filtered_comparison.iterrows():
        metric_name_lower = row["metric_name"].lower()
        category = metric_categories.get(metric_name_lower, "Other")
        scenario = row["scenario"]

        # Show scenario header when viewing all scenarios and scenario changes
        if selected_detail_scenario == "All Scenarios" and scenario != current_scenario:
            current_scenario = scenario
            current_category = None  # Reset category for new scenario
            st.markdown(f'<div style="margin-top: 1.5rem; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid {COLORS["navy"]}; font-weight: 600; color: {COLORS["navy"]};">{scenario}</div>', unsafe_allow_html=True)

        # Show category header when category changes
        if category != current_category:
            current_category = category
            st.markdown(f'<div style="margin-top: 0.75rem; margin-bottom: 0.5rem; font-size: 0.8rem; font-weight: 500; color: {COLORS["medium_gray"]}; text-transform: uppercase; letter-spacing: 0.05em;">{category}</div>', unsafe_allow_html=True)
        delta = row["delta"]
        pct = row["delta_pct"]
        baseline = row["metric_value_baseline"]
        compare = row["metric_value_compare"]

        # Check if this metric is "lower is better"
        is_lower_better = metric_name_lower in lower_is_better

        # Determine if this change is good or bad
        if abs(delta) <= 0.001:
            # No significant change
            change_color = COLORS['medium_gray']
            change_text = "— No change"
            border_class = ""
        elif delta > 0:
            # Value increased
            if is_lower_better:
                # Increase in bias/mae/rmse = BAD
                change_color = COLORS['poor']
                border_class = "status-poor"
            else:
                # Increase in f1/precision/recall = GOOD
                change_color = COLORS['good']
                border_class = "status-good"
            change_text = f"↑ +{delta:.3f} (+{pct:.1f}%)"
        else:
            # Value decreased
            if is_lower_better:
                # Decrease in bias/mae/rmse = GOOD
                change_color = COLORS['good']
                border_class = "status-good"
            else:
                # Decrease in f1/precision/recall = BAD
                change_color = COLORS['poor']
                border_class = "status-poor"
            change_text = f"↓ {delta:.3f} ({pct:.1f}%)"

        # Build HTML without multi-line f-string to avoid parsing issues
        card_html = (
            f'<div class="metric-card {border_class}" style="margin-bottom: 0.5rem; padding: 0.75rem 1rem;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div><span style="color: {COLORS["navy"]}; font-weight: 500;">{row["metric_name"]}</span></div>'
            f'<span style="color: {change_color}; font-weight: 500; font-size: 0.9rem;">{change_text}</span>'
            f'</div>'
            f'<div style="display: flex; gap: 2rem; margin-top: 0.5rem; font-size: 0.85rem;">'
            f'<span style="color: {COLORS["medium_gray"]};">Baseline: <strong style="color: {COLORS["charcoal"]}; font-family: monospace;">{baseline:.3f}</strong></span>'
            f'<span style="color: {COLORS["medium_gray"]};">Compare: <strong style="color: {COLORS["charcoal"]}; font-family: monospace;">{compare:.3f}</strong></span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Visual comparison chart
    render_section_header("Visual Comparison")

    scenarios = comparison_df["scenario"].unique()
    if len(scenarios) > 0:
        selected_scenario = st.selectbox("Select scenario", scenarios, key="compare_scenario")

        scenario_data = comparison_df[comparison_df["scenario"] == selected_scenario]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Baseline",
            x=scenario_data["metric_name"],
            y=scenario_data["metric_value_baseline"],
            marker_color=COLORS["medium_gray"]
        ))

        fig.add_trace(go.Bar(
            name="Compare",
            x=scenario_data["metric_name"],
            y=scenario_data["metric_value_compare"],
            marker_color=COLORS["teal"]
        ))

        fig.update_layout(
            title=f"Metrics Comparison: {selected_scenario}",
            barmode="group",
            height=400,
            showlegend=True
        )

        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Insights
    render_section_header("Insights")

    # Categorize significant changes accounting for metric direction
    significant_improvements = []
    significant_declines = []

    # Use filtered data to respect scenario selection
    for _, row in filtered_comparison.iterrows():
        delta = row["delta"]
        metric_name = row["metric_name"].lower()
        is_lower_better = metric_name in lower_is_better_metrics

        # Check if change is significant (>5%)
        if abs(delta) < 0.05:
            continue

        is_improvement = (delta < 0 and is_lower_better) or (delta > 0 and not is_lower_better)

        if is_improvement:
            significant_improvements.append(row)
        else:
            significant_declines.append(row)

    if significant_declines:
        decline_items = ""
        for row in significant_declines:
            direction = "↑" if row['delta'] > 0 else "↓"
            scenario_text = f' ({row["scenario"]})' if selected_detail_scenario == "All Scenarios" else ""
            decline_items += f'<div style="margin-bottom: 0.25rem;">• <strong>{row["metric_name"]}</strong>{scenario_text}: {row["metric_value_baseline"]:.3f} → {row["metric_value_compare"]:.3f} ({direction} {row["delta_pct"]:.1f}%)</div>'

        decline_html = (
            f'<div class="metric-card status-poor">'
            f'<span class="metric-label">Significant Declines Detected</span>'
            f'<div style="margin-top: 0.75rem; color: {COLORS["charcoal"]};">{decline_items}</div>'
            f'<div class="metric-context">Consider reviewing recent changes to prompts, test data, or model configuration.</div>'
            f'</div>'
        )
        st.markdown(decline_html, unsafe_allow_html=True)

    if significant_improvements:
        improvement_items = ""
        for row in significant_improvements:
            direction = "↑" if row['delta'] > 0 else "↓"
            scenario_text = f' ({row["scenario"]})' if selected_detail_scenario == "All Scenarios" else ""
            improvement_items += f'<div style="margin-bottom: 0.25rem;">• <strong>{row["metric_name"]}</strong>{scenario_text}: {row["metric_value_baseline"]:.3f} → {row["metric_value_compare"]:.3f} ({direction} {row["delta_pct"]:.1f}%)</div>'

        improvement_html = (
            f'<div class="metric-card status-good">'
            f'<span class="metric-label">Significant Improvements</span>'
            f'<div style="margin-top: 0.75rem; color: {COLORS["charcoal"]};">{improvement_items}</div>'
            f'</div>'
        )
        st.markdown(improvement_html, unsafe_allow_html=True)

    if not significant_declines and not significant_improvements:
        stable_html = (
            f'<div class="metric-card">'
            f'<span class="metric-label">Stable Performance</span>'
            f'<div style="margin-top: 0.5rem; color: {COLORS["charcoal"]};">No significant changes detected between these runs. Performance is consistent.</div>'
            f'</div>'
        )
        st.markdown(stable_html, unsafe_allow_html=True)


# ============================================
# PAGE: RUN EVALUATION
# ============================================

def load_config():
    """Load configuration from settings.yaml."""
    import yaml
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None


def render_run_evaluation_page():
    """Page for running new evaluations."""
    import os

    render_page_header(
        "Run Evaluation",
        "Execute hallucination detection evaluations"
    )

    # Password protection
    render_section_header("Authentication")

    # Get admin password from secrets or use default
    try:
        admin_password = st.secrets.get("ADMIN_PASSWORD", "1978")
    except Exception:
        admin_password = "1978"

    # Check if already authenticated in session
    if "eval_authenticated" not in st.session_state:
        st.session_state.eval_authenticated = False

    if not st.session_state.eval_authenticated:
        password_input = st.text_input("Enter admin password to run evaluations", type="password")

        if password_input:
            if password_input == admin_password:
                st.session_state.eval_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return

    # Show authenticated status
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span style="color: {COLORS['good']};">✓</span>
        <span style="color: {COLORS['medium_gray']}; font-size: 0.85rem;">Authenticated</span>
    </div>
    """, unsafe_allow_html=True)

    # Load config
    config = load_config()

    if config is None:
        st.error("Could not load config/settings.yaml. Make sure the file exists.")
        return

    # Check for API key - check both env and Streamlit secrets
    api_key = os.getenv("OPENAI_API_KEY")

    # Try to get from Streamlit secrets if not in env
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                # Set it in environment so orchestrator can use it
                os.environ["OPENAI_API_KEY"] = api_key
        except Exception:
            pass

    # Show API key status
    render_section_header("System Status")

    if api_key:
        # Show masked key
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 15 else "***"
        st.markdown(f"""
        <div class="metric-card status-good" style="padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: {COLORS['good']}; font-size: 1.2rem;">✓</span>
                <span style="color: {COLORS['charcoal']};">OpenAI API Key configured</span>
            </div>
            <div style="font-family: monospace; font-size: 0.8rem; color: {COLORS['medium_gray']}; margin-top: 0.5rem;">
                {masked_key}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card status-poor" style="padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: {COLORS['poor']}; font-size: 1.2rem;">✗</span>
                <span style="color: {COLORS['charcoal']};">OpenAI API Key missing</span>
            </div>
            <div style="font-size: 0.85rem; color: {COLORS['medium_gray']}; margin-top: 0.5rem;">
                Add OPENAI_API_KEY to Streamlit secrets (Settings → Secrets)
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Configuration section
    render_section_header("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Model selection
        available_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        default_model = config.get('model') or config.get('global', {}).get('default_model', 'gpt-4o-mini')

        model = st.selectbox(
            "Model",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            help="LLM model to use for evaluation"
        )

    with col2:
        # Sample size
        default_sample = config.get('global', {}).get('sample_size', 20)
        sample_size = st.number_input(
            "Sample Size",
            min_value=5,
            max_value=100,
            value=default_sample,
            step=5,
            help="Number of test cases to evaluate"
        )

    # Scenario selection
    scenarios_config = config.get('scenarios', {})
    available_scenarios = list(scenarios_config.keys())
    enabled_by_default = [s for s, cfg in scenarios_config.items() if cfg.get('enabled', False)]

    selected_scenarios = st.multiselect(
        "Scenarios",
        available_scenarios,
        default=enabled_by_default,
        help="Select evaluation scenarios to run"
    )

    st.markdown("---")

    # Run controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        dry_run = st.checkbox("Dry Run", help="Preview what would run without executing")

    with col2:
        run_button = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    # Results area
    if run_button:
        if not selected_scenarios:
            st.error("Please select at least one scenario.")
            return

        render_section_header("Execution")

        if dry_run:
            # Dry run - just show what would happen
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Dry Run Preview</span>
                <div style="margin-top: 0.75rem; color: {COLORS['charcoal']};">
                    <p><strong>Model:</strong> {model}</p>
                    <p><strong>Sample Size:</strong> {sample_size}</p>
                    <p><strong>Scenarios:</strong></p>
                    <ul>
            """, unsafe_allow_html=True)

            for scenario in selected_scenarios:
                prompt_version = scenarios_config.get(scenario, {}).get('prompt_version', 'v1_zero_shot')
                st.markdown(f"<li>{scenario} (prompt: {prompt_version})</li>", unsafe_allow_html=True)

            st.markdown("""
                    </ul>
                </div>
                <div class="metric-context">
                    Uncheck "Dry Run" and click Run Evaluation to execute.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Actually run the evaluation
            try:
                # Import orchestrator with absolute path
                import sys
                import importlib
                from pathlib import Path

                # Use absolute path to src directory
                src_dir = Path(__file__).parent
                if str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))

                # Force reload of modules to pick up code changes
                import evaluator as eval_module
                import orchestrator as orch_module
                importlib.reload(eval_module)
                importlib.reload(orch_module)
                from orchestrator import EvaluationOrchestrator

                with st.spinner("Running evaluation... This may take a few minutes."):
                    # Progress placeholder
                    progress_placeholder = st.empty()
                    results_placeholder = st.empty()

                    # Debug: Show what we're about to run
                    st.info(f"Starting evaluation: model={model}, sample_size={sample_size}, scenarios={selected_scenarios}")

                    # Run evaluation
                    orch_instance = EvaluationOrchestrator()
                    summary = orch_instance.run_daily_evaluation(
                        scenarios=selected_scenarios,
                        model=model,
                        sample_size=sample_size,
                        dry_run=False
                    )

                    # Debug: Show what we got back
                    if summary:
                        st.info(f"Run completed: {summary.run_id}, passed={summary.scenarios_passed}, failed={summary.scenarios_failed}")

                if summary:
                    # Show results
                    status_color = COLORS['good'] if summary.overall_status.value == 'healthy' else (
                        COLORS['poor'] if summary.overall_status.value == 'critical' else COLORS['amber']
                    )
                    status_class = "status-good" if summary.overall_status.value == 'healthy' else (
                        "status-poor" if summary.overall_status.value == 'critical' else "status-warning"
                    )

                    st.markdown(f"""
                    <div class="metric-card {status_class}">
                        <span class="metric-label">Evaluation Complete</span>
                        <div style="margin-top: 0.75rem;">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                                {summary.overall_status.value.upper()}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Summary cards
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="metric-label">Run ID</span>
                            <div style="font-family: monospace; margin-top: 0.5rem; color: {COLORS['navy']};">
                                {summary.run_id}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card status-good">
                            <span class="metric-label">Passed</span>
                            <div class="metric-value" style="color: {COLORS['good']};">{summary.scenarios_passed}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        fail_class = "status-poor" if summary.scenarios_failed > 0 else ""
                        fail_color = COLORS['poor'] if summary.scenarios_failed > 0 else COLORS['charcoal']
                        st.markdown(f"""
                        <div class="metric-card {fail_class}">
                            <span class="metric-label">Failed</span>
                            <div class="metric-value" style="color: {fail_color};">{summary.scenarios_failed}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Alerts
                    if summary.alerts:
                        render_section_header("Alerts")
                        for alert in summary.alerts:
                            st.markdown(f"""
                            <div class="metric-card status-warning" style="margin-bottom: 0.5rem;">
                                <span style="color: {COLORS['amber']};">⚠️ {alert}</span>
                            </div>
                            """, unsafe_allow_html=True)

                    st.success("✓ Results saved to database. View them in Metrics Overview or Trends.")

                else:
                    st.error("Evaluation failed. Check the logs for details.")

            except Exception as e:
                st.error(f"Error running evaluation: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    # Recent runs info
    render_section_header("Recent Runs")

    try:
        run_ids = db.get_run_ids()
        if run_ids:
            st.caption(f"Last 5 runs:")
            for run_id in run_ids[:5]:
                st.markdown(f"• `{run_id}`")
        else:
            st.caption("No runs recorded yet.")
    except Exception:
        st.caption("Could not load run history.")


# ============================================
# PAGE: METRIC GUIDE
# ============================================

def render_guide_page():
    """Educational guide to metrics."""

    render_page_header(
        "Understanding Metrics",
        "A practical guide to AI evaluation metrics"
    )

    # Confusion Matrix primer
    render_section_header("The Foundation: Confusion Matrix")

    st.markdown(f"""
    Every classification metric builds on four fundamental counts:

    |  | **Predicted: Hallucination** | **Predicted: Grounded** |
    |--|------------------------------|-------------------------|
    | **Actual: Hallucination** | TP (True Positive) | FN (False Negative) |
    | **Actual: Grounded** | FP (False Positive) | TN (True Negative) |

    - **TP**: Correctly flagged a hallucination
    - **TN**: Correctly accepted grounded content
    - **FP**: Wrongly flagged grounded content (false alarm)
    - **FN**: Missed an actual hallucination (dangerous)
    """)

    st.markdown("---")

    # Classification metrics
    render_section_header("Classification Metrics")

    with st.expander("**Precision** — Trust in Alerts", expanded=True):
        st.markdown(f"""
        **Formula:** TP / (TP + FP)

        **Question answered:** When the model flags something as a hallucination, how often is it right?

        **Intuition:** High precision means your alerts are trustworthy. When users see a warning,
        they can believe it. Low precision leads to "alert fatigue" — users ignore warnings because
        they're often wrong.

        **Target:** ≥ 0.75 for production use
        """)

    with st.expander("**Recall** — Catch Rate", expanded=True):
        st.markdown(f"""
        **Formula:** TP / (TP + FN)

        **Question answered:** Of all actual hallucinations, how many did we catch?

        **Intuition:** High recall means few hallucinations slip through. Critical for safety-sensitive
        applications where missing a hallucination could cause real harm (medical, legal, financial).

        **Target:** ≥ 0.75 for most applications, higher for safety-critical
        """)

    with st.expander("**F1 Score** — Balanced Performance", expanded=True):
        st.markdown(f"""
        **Formula:** 2 × (Precision × Recall) / (Precision + Recall)

        **Question answered:** How well does the model balance catching hallucinations vs. avoiding false alarms?

        **Intuition:** The harmonic mean punishes extreme imbalances. You can't game F1 by optimizing
        only precision or only recall. It's the primary metric for overall detection quality.

        **Target:** ≥ 0.75 indicates production-ready performance
        """)

    st.markdown("---")

    # Agreement metrics
    render_section_header("Agreement Metrics")

    with st.expander("**Cohen's Kappa** — Chance-Corrected Agreement", expanded=True):
        st.markdown(f"""
        **Formula:** (Observed Agreement - Chance Agreement) / (1 - Chance Agreement)

        **Question answered:** How much better than random guessing is this model?

        **Intuition:** Accuracy can be misleading with imbalanced classes. Kappa corrects for chance —
        a Kappa of 0 means no better than random, while 1 means perfect agreement.

        **Interpretation scale:**
        - 0.81-1.00: Almost perfect
        - 0.61-0.80: Substantial
        - 0.41-0.60: Moderate
        - 0.21-0.40: Fair
        - 0.00-0.20: Slight
        """)

    st.markdown("---")

    # Correlation metrics
    render_section_header("Correlation Metrics")

    st.markdown("""
    These metrics measure how well **confidence scores** predict **actual correctness**.
    A well-calibrated model should be more confident when it's right.
    """)

    with st.expander("**Kendall's Tau** — Rank Agreement"):
        st.markdown(f"""
        **Question answered:** When comparing any two predictions, does higher confidence usually mean higher correctness?

        **Intuition:** Counts concordant vs discordant pairs. More robust to ties and outliers than other
        correlation measures. Good for small sample sizes.

        **Target:** ≥ 0.60 (Tau values are inherently smaller than Pearson/Spearman)
        """)

    with st.expander("**Pearson Correlation** — Linear Relationship"):
        st.markdown(f"""
        **Question answered:** Is there a proportional relationship between confidence and correctness?

        **Intuition:** If confidence of 0.8 means 80% correct, that's perfect linear calibration.
        Sensitive to outliers.

        **Target:** ≥ 0.70
        """)

    st.markdown("---")

    # Which metric to prioritize
    render_section_header("Which Metric Should You Prioritize?")

    st.markdown(f"""
    | Your Situation | Primary Metric | Why |
    |---------------|----------------|-----|
    | General evaluation | **F1 Score** | Balanced measure of overall quality |
    | Safety-critical (medical, legal) | **Recall** | Missing hallucinations is dangerous |
    | User-facing alerts | **Precision + TNR** | False alarms erode trust |
    | Comparing model versions | **F1** for ranking, **Kappa** for consistency |
    | Calibration analysis | **Kendall's Tau** or **Pearson** | Confidence should predict correctness |
    """)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.set_page_config(
        page_title="Eval Lab · rasar.ai",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply brand styling
    apply_brand_css()

    # Load data
    df = load_metrics()

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-size: 1.1rem; font-weight: 500; color: {COLORS['navy']};">
                ◈ Eval Lab
            </div>
            <div style="font-size: 0.75rem; color: {COLORS['medium_gray']}; margin-top: 0.25rem;">
                learn AI evaluation
            </div>
            <div style="font-size: 0.7rem; margin-top: 0.25rem;">
                <a href="https://rasar.ai" target="_blank" style="color: {COLORS['teal']};">rasar.ai</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            ["Metrics Overview", "Trends", "Compare Runs", "Run History", "Run Evaluation", "Understanding Metrics"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        if not df.empty:
            total_runs = df["run_id"].nunique()
            st.caption(f"**{total_runs}** evaluation runs")

            latest_run = df.sort_values("timestamp", ascending=False).iloc[0]
            st.caption(f"Latest: {latest_run['timestamp'][:10]}")

        # Database info (collapsed)
        with st.expander("System", expanded=False):
            try:
                debug = db.debug_info()
                st.caption(f"Backend: {debug.get('backend', 'Unknown')}")
                st.caption(f"Metrics: {debug.get('metrics_count', 0)}")
                st.caption(f"Test Results: {debug.get('test_results_count', 0)}")
            except Exception as e:
                st.caption(f"Error: {e}")

    # Route to pages
    if page == "Metrics Overview":
        render_metrics_overview_page(df)
    elif page == "Trends":
        render_trends_page(df)
    elif page == "Compare Runs":
        render_compare_runs_page(df)
    elif page == "Run History":
        render_run_history_page(df)
    elif page == "Run Evaluation":
        render_run_evaluation_page()
    elif page == "Understanding Metrics":
        render_guide_page()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
