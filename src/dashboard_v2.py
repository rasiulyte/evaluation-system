"""
LLM Evaluation Dashboard - rasar.ai
A thoughtful, minimal dashboard for AI evaluation metrics.

Design Philosophy: Intellectual, understated, quietly confident.
Technical depth without pretension. Quality over flash.
"""

# Version for debugging - update this when making changes
DASHBOARD_VERSION = "1.5.0"

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from datetime import datetime, timedelta

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
# TIMEZONE HELPERS
# ============================================

def to_pst(timestamp_str: str) -> tuple:
    """
    Convert UTC timestamp to PST for display.
    Streamlit Cloud stores timestamps in UTC, subtract 8 hours for PST.

    Args:
        timestamp_str: ISO format timestamp in UTC (e.g., "2026-01-27T03:14:37")

    Returns:
        (date_str, time_str) in PST
    """
    if not timestamp_str or len(timestamp_str) < 10:
        return ("—", "—")

    try:
        # Parse the timestamp
        dt = datetime.fromisoformat(timestamp_str.split('.')[0])  # Remove microseconds

        # Convert UTC to PST (subtract 8 hours)
        pst_dt = dt - timedelta(hours=8)

        return (pst_dt.strftime("%Y-%m-%d"), pst_dt.strftime("%H:%M"))
    except Exception:
        # Fallback to raw string parsing
        return (timestamp_str[:10], timestamp_str[11:16] if len(timestamp_str) > 16 else "—")


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

    /* Logo button - styled as text, left aligned, LARGER */
    [data-testid="stSidebar"] .stButton:first-of-type {{
        text-align: left !important;
        margin-bottom: 0.25rem !important;
    }}

    [data-testid="stSidebar"] .stButton:first-of-type > button {{
        background-color: transparent !important;
        color: {COLORS['navy']} !important;
        border: none !important;
        padding: 0.75rem 0 !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: auto !important;
        min-width: 0 !important;
        letter-spacing: -0.01em !important;
    }}

    [data-testid="stSidebar"] .stButton:first-of-type > button:hover {{
        background-color: transparent !important;
        color: {COLORS['teal']} !important;
    }}

    /* ===== SIDEBAR NAVIGATION - Link Style with Teal Circle Indicator ===== */

    /* Navigation buttons as links - default (inactive) state */
    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button {{
        background: transparent !important;
        border: none !important;
        color: {COLORS['charcoal']} !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        padding: 4px 8px !important;
        margin: 0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        min-height: 0 !important;
        height: auto !important;
        cursor: pointer !important;
    }}

    /* Active page (primary button) - teal colored */
    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button[kind="primary"],
    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button[data-testid="stBaseButton-primary"] {{
        background: transparent !important;
        border: none !important;
        color: {COLORS['teal']} !important;
        font-weight: 500 !important;
    }}

    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button:hover {{
        background: transparent !important;
        color: {COLORS['teal']} !important;
    }}

    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button:focus {{
        box-shadow: none !important;
        outline: none !important;
    }}

    [data-testid="stSidebar"] .stButton:not(:first-of-type) > button p {{
        margin: 0 !important;
        line-height: 1.4 !important;
    }}

    /* Section headers */
    .nav-section-header {{
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {COLORS['navy']} !important;
        margin: 20px 0 10px 0 !important;
        padding: 8px 12px !important;
        background: {COLORS['light_gray']} !important;
        border-radius: 6px !important;
        border-left: 3px solid {COLORS['teal']} !important;
    }}

    .nav-section-header.learn {{
        border-left-color: #8b5cf6 !important;
        background: linear-gradient(90deg, #8b5cf615, {COLORS['light_gray']}) !important;
    }}

    .nav-section-header.analyze {{
        border-left-color: #f59e0b !important;
        background: linear-gradient(90deg, #f59e0b15, {COLORS['light_gray']}) !important;
    }}

    .nav-section-header.run {{
        border-left-color: #10b981 !important;
        background: linear-gradient(90deg, #10b98115, {COLORS['light_gray']}) !important;
    }}

    /* Bottom metadata area */
    .nav-metadata {{
        border-top: 1px solid #e5e7eb !important;
        padding-top: 16px !important;
        margin-top: 24px !important;
        font-size: 13px !important;
        color: #6b7280 !important;
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

    /* Tablet */
    @media (max-width: 1024px) {{
        .main .block-container {{
            padding: 1.5rem 2rem;
        }}

        .metric-card {{
            padding: 0.875rem;
        }}
    }}

    /* Mobile */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem 1rem;
        }}

        .metric-value {{
            font-size: 1.5rem;
        }}

        .metric-card {{
            padding: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .section-header {{
            font-size: 1.25rem;
        }}

        .page-header h1 {{
            font-size: 1.5rem;
        }}

        /* Stack columns on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}

    /* Small mobile */
    @media (max-width: 480px) {{
        .main .block-container {{
            padding: 0.75rem 0.5rem;
        }}

        .metric-value {{
            font-size: 1.25rem;
        }}

        .metric-label {{
            font-size: 0.65rem;
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
        <p style="margin-top: 0.5rem;">&copy; 2025 <a href="https://rasar.ai" target="_blank">rasar.ai</a>. All rights reserved.</p>
        <p style="margin-top: 0.25rem; font-size: 0.7rem; color: {COLORS['medium_gray']};">v{DASHBOARD_VERSION}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# DATA LOADING
# ============================================

def load_test_case(test_case_id: str) -> dict:
    """Load a test case by ID from the test_cases directory."""
    import json
    from pathlib import Path

    # Try to find the test case in JSON files
    project_root = Path(__file__).parent.parent
    test_cases_dir = project_root / "data" / "test_cases"

    if not test_cases_dir.exists():
        return None

    for json_file in test_cases_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                cases = json.load(f)
                for case in cases:
                    if case.get("id") == test_case_id:
                        return case
        except Exception:
            continue

    return None


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_metrics() -> pd.DataFrame:
    """Load metrics from database."""
    try:
        metrics = db.get_all_metrics()
        if metrics:
            return pd.DataFrame(metrics)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_daily_runs() -> dict:
    """Load daily runs from database as a dict keyed by run_id."""
    try:
        runs = db.get_daily_runs()
        return {r['run_id']: r for r in runs} if runs else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)  # Cache for 60 seconds
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
        st.info("No evaluation data yet. Go to **Run Evaluation** in the sidebar to run your first evaluation.")
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

    run_date, run_time = to_pst(latest_run['timestamp'])
    st.caption(f"Latest run: {latest_run['run_id']} · {run_date} {run_time}")

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
        # Order metrics logically: classification, agreement, correlation, calibration
        preferred_order = ["f1", "precision", "recall", "tnr", "accuracy", "cohens_kappa",
                          "spearman", "pearson", "kendalls_tau", "bias", "mae", "rmse"]
        metric_options = [m for m in preferred_order if m in available_metrics]
        # Add any metrics not in preferred order
        metric_options += [m for m in available_metrics if m not in metric_options]
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
        st.info("No runs recorded yet. Go to **Run Evaluation** to create your first run.")
        return

    # Fetch daily runs for cost info (cached)
    daily_runs = load_daily_runs()

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

        # Get cost info from daily_runs
        daily_run = daily_runs.get(run_id, {})
        total_tokens = daily_run.get('total_tokens', 0) or 0
        cost_usd = daily_run.get('total_cost_usd', 0.0) or 0.0

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

        # Format timestamp for display
        date_str, time_str = to_pst(timestamp)

        runs_data.append({
            "Run ID": run_id,
            "Date": date_str,
            "Time": time_str,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Status": status,
            "status_raw": status_raw,
            "tokens": total_tokens,
            "cost": cost_usd
        })

    runs_df = pd.DataFrame(runs_data).sort_values("Date", ascending=False)

    # Summary cards with red/green
    passing_runs = len(runs_df[runs_df["status_raw"] == "passing"])
    failing_runs = len(runs_df[runs_df["status_raw"] == "failing"])
    total_cost = runs_df["cost"].sum() if "cost" in runs_df.columns else 0.0
    total_tokens = runs_df["tokens"].sum() if "tokens" in runs_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_simple_metric("Total Runs", str(len(runs_df)))
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Pass / Fail</span>
            <div class="metric-value">
                <span style="color: {COLORS['good']};">{passing_runs}</span>
                <span style="color: {COLORS['medium_gray']}; font-size: 1rem;"> / </span>
                <span style="color: {COLORS['poor']};">{failing_runs}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Total Cost</span>
            <div class="metric-value" style="color: {COLORS['teal']}; font-size: 1.25rem;">${total_cost:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Total Tokens</span>
            <div class="metric-value" style="font-size: 1.1rem;">{total_tokens:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Helper function to render a single run
    def render_run_card(row):
        status_raw = row['status_raw']
        if status_raw == "passing":
            status_color = COLORS['good']
            status_text = "✓ Passing"
        elif status_raw == "failing":
            status_color = COLORS['poor']
            status_text = "✗ Failing"
        else:
            status_color = COLORS['amber']
            status_text = "○ Fair"

        f1_str = f"{row['F1']:.3f}" if row['F1'] is not None else "—"
        prec_str = f"{row['Precision']:.3f}" if row['Precision'] is not None else "—"
        rec_str = f"{row['Recall']:.3f}" if row['Recall'] is not None else "—"
        cost_str = f"${row['cost']:.4f}"  # Always show cost, even if $0.0000
        tokens_str = f"{row['tokens']:,}" if row['tokens'] > 0 else "0"

        # Create expander for each run
        with st.expander(f"**{row['Run ID']}** — {row['Date']} {row['Time']} — F1: {f1_str} — Cost: {cost_str}"):
            # Summary row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"**Status:** <span style='color: {status_color};'>{status_text}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**F1:** {f1_str}")
            with col3:
                st.markdown(f"**Precision:** {prec_str}")
            with col4:
                st.markdown(f"**Recall:** {rec_str}")
            with col5:
                st.markdown(f"**Tokens:** {tokens_str}")

            # Show note if cost is zero
            if row['cost'] == 0:
                st.caption("💡 Cost is $0 because this run was recorded before cost tracking was added.")

            st.markdown("---")

            # Fetch test results for this run
            try:
                test_results = db.get_test_results(run_id=row['Run ID'])
                if test_results:
                    # Group by scenario
                    scenarios = {}
                    for result in test_results:
                        scenario = result.get('scenario', 'unknown')
                        if scenario not in scenarios:
                            scenarios[scenario] = []
                        scenarios[scenario].append(result)

                    # Display by scenario
                    for scenario, results in sorted(scenarios.items()):
                        st.markdown(f"**Scenario: {scenario}** ({len(results)} test cases)")

                        # Build results table
                        table_data = []
                        for r in results:
                            prediction = r.get('prediction', '—')
                            ground_truth = r.get('ground_truth', '—')
                            correct = "✓" if r.get('correct') else "✗"
                            confidence = r.get('confidence', 0)
                            conf_str = f"{confidence:.2f}" if confidence else "—"

                            table_data.append({
                                "Test Case": r.get('test_case_id', '—'),
                                "Prompt": r.get('prompt_id', '—'),
                                "Prediction": prediction,
                                "Ground Truth": ground_truth,
                                "Correct": correct,
                                "Confidence": conf_str
                            })

                        if table_data:
                            table_df = pd.DataFrame(table_data)
                            st.dataframe(table_df, use_container_width=True, hide_index=True)

                            # Allow viewing individual test cases
                            test_case_ids = [r.get('test_case_id') for r in results if r.get('test_case_id')]
                            if test_case_ids:
                                selected_case = st.selectbox(
                                    "View test case details:",
                                    ["Select a test case..."] + test_case_ids,
                                    key=f"tc_{row['Run ID']}_{scenario}"
                                )

                                if selected_case and selected_case != "Select a test case...":
                                    # Load test case from file
                                    test_case_data = load_test_case(selected_case)

                                    # Find the LLM result for this test case
                                    llm_result = next((r for r in results if r.get('test_case_id') == selected_case), None)

                                    if test_case_data:
                                        st.markdown(f"---")
                                        st.markdown(f"##### Test Case: `{selected_case}`")

                                        # Status row
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            expected_label = test_case_data.get('label', '—')
                                            label_color = COLORS['good'] if expected_label == 'grounded' else COLORS['poor']
                                            st.markdown(f"**Expected Label:** <span style='color: {label_color};'>`{expected_label}`</span>", unsafe_allow_html=True)
                                        with col2:
                                            if llm_result:
                                                pred = llm_result.get('prediction', '—')
                                                pred_color = COLORS['good'] if pred == 'grounded' else COLORS['poor']
                                                st.markdown(f"**LLM Prediction:** <span style='color: {pred_color};'>`{pred}`</span>", unsafe_allow_html=True)
                                        with col3:
                                            if llm_result:
                                                is_correct = llm_result.get('correct', False)
                                                result_text = "✓ Correct" if is_correct else "✗ Wrong"
                                                result_color = COLORS['good'] if is_correct else COLORS['poor']
                                                st.markdown(f"**Result:** <span style='color: {result_color};'>{result_text}</span>", unsafe_allow_html=True)

                                        if test_case_data.get('failure_mode'):
                                            st.markdown(f"**Failure Mode:** `{test_case_data.get('failure_mode')}`")

                                        st.markdown("---")

                                        # Context and Response
                                        st.markdown("**Context:**")
                                        context_text = test_case_data.get('context', '—')
                                        st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.85rem;">{context_text}</div>""", unsafe_allow_html=True)

                                        st.markdown("**Response:**")
                                        response_text = test_case_data.get('response', '—')
                                        st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.85rem;">{response_text}</div>""", unsafe_allow_html=True)

                                        # LLM's analysis
                                        if llm_result:
                                            st.markdown("---")
                                            st.markdown("**LLM Analysis:**")

                                            # Confidence
                                            conf = llm_result.get('confidence', 0)
                                            conf_pct = conf * 100 if conf else 0
                                            st.markdown(f"**Confidence:** `{conf_pct:.0f}%`")

                                            # Parse and display LLM output
                                            llm_output = llm_result.get('llm_output', '')

                                            # Try to parse JSON for nicer display
                                            try:
                                                import json
                                                llm_json = json.loads(llm_output)

                                                st.markdown("**Classification:**")
                                                classification = llm_json.get('classification', '—')
                                                class_color = COLORS['good'] if classification == 'grounded' else COLORS['poor']
                                                st.markdown(f"<span style='color: {class_color}; font-weight: bold;'>{classification}</span>", unsafe_allow_html=True)

                                                st.markdown("**LLM Reasoning:**")
                                                reasoning = llm_json.get('reasoning', '—')
                                                st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.85rem; border-left: 4px solid {COLORS['teal']};">{reasoning}</div>""", unsafe_allow_html=True)
                                            except:
                                                # Show raw output if not JSON
                                                st.markdown("**Raw LLM Output:**")
                                                st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.85rem; border-left: 4px solid {COLORS['teal']}; white-space: pre-wrap;">{llm_output}</div>""", unsafe_allow_html=True)

                                            # Additional metadata
                                            st.markdown(f"<small style='color: {COLORS['medium_gray']};'>Model: {llm_result.get('model', '—')} | Duration: {llm_result.get('duration_ms', 0):.0f}ms</small>", unsafe_allow_html=True)
                                        else:
                                            st.info("No LLM response found for this test case.")
                                    else:
                                        st.warning(f"Could not load test case: {selected_case}")

                        st.markdown("")
                else:
                    st.info("No detailed test results available for this run.")
            except Exception as e:
                st.warning(f"Could not load test results: {e}")

    # Display first 10 runs
    st.markdown(f"### Recent Runs")
    for idx, row in runs_df.head(10).iterrows():
        render_run_card(row)

    # Show older runs in expander if there are more than 10
    if len(runs_df) > 10:
        older_runs = runs_df.iloc[10:]
        with st.expander(f"Show {len(older_runs)} older runs"):
            for idx, row in older_runs.iterrows():
                render_run_card(row)


# ============================================
# PAGE: SLICE ANALYSIS
# ============================================

def render_slice_analysis_page(df: pd.DataFrame):
    """Analyze performance across different slices (failure modes, difficulty, etc.)."""

    render_page_header(
        "Slice Analysis",
        "See how performance varies across different subsets of test cases"
    )

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <a href="https://github.com/rasiulyte/evaluation-system/blob/main/docs/SLICE_ANALYSIS.md" target="_blank"
           style="color: {COLORS['teal']}; text-decoration: none; font-size: 0.9rem;">
            📄 View full documentation on GitHub →
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Explanation
    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem; margin-bottom: 1.5rem;">
        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">What is Slice Analysis?</div>
        <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
            Think of it like a report card that shows grades for each subject, not just the overall GPA.
            Your model might score 85% overall, but struggle with specific types of problems.
            Slice analysis reveals these hidden weaknesses by breaking down performance into categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get test results with failure mode info
    try:
        test_results = db.get_test_results()
        if not test_results:
            st.info("No test results available. Go to **Run Evaluation** to generate test results.")
            return

        results_df = pd.DataFrame(test_results)

        # Load test case metadata to get failure_mode and difficulty
        all_test_cases = load_all_test_cases()
        test_case_meta = {tc['id']: tc for tc in all_test_cases}

        # Enrich results with metadata
        results_df['failure_mode'] = results_df['test_case_id'].apply(
            lambda x: test_case_meta.get(x, {}).get('failure_mode', 'unknown')
        )
        results_df['difficulty'] = results_df['test_case_id'].apply(
            lambda x: test_case_meta.get(x, {}).get('difficulty', 'unknown')
        )

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Run selector
    runs = results_df['run_id'].unique().tolist() if 'run_id' in results_df.columns else []
    if not runs:
        st.info("No runs available for slice analysis.")
        return

    selected_run = st.selectbox(
        "Select Run to Analyze",
        runs,
        index=0,
        key="slice_run_select"
    )

    run_data = results_df[results_df['run_id'] == selected_run] if 'run_id' in results_df.columns else results_df

    st.markdown("---")

    # ==========================================
    # SLICE BY FAILURE MODE
    # ==========================================
    render_section_header("By Failure Mode")

    # Beginner-friendly explanation
    st.markdown(f"""
    <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 3px solid {COLORS['teal']};">
        <div style="font-size: 0.85rem; color: {COLORS['charcoal']}; line-height: 1.6;">
            <strong>How to read this:</strong> Each card shows how well the model handles a specific type of test case.
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.25rem;">
                <li><strong>Green (75%+)</strong> = Good performance</li>
                <li><strong>Yellow (60-74%)</strong> = Needs improvement</li>
                <li><strong>Red (&lt;60%)</strong> = Problem area to focus on</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    failure_modes = run_data['failure_mode'].unique()
    failure_mode_stats = []

    # Failure modes that test "grounded" cases (no hallucinations to detect)
    grounded_failure_modes = {'factual_addition', 'valid_inference', 'verbatim_grounded'}

    for fm in sorted(failure_modes):
        if fm == 'unknown':
            continue
        fm_data = run_data[run_data['failure_mode'] == fm]
        total = len(fm_data)
        correct = fm_data['correct'].sum() if 'correct' in fm_data.columns else 0
        accuracy = correct / total if total > 0 else 0

        # Check if this failure mode contains hallucination cases
        is_grounded_mode = fm in grounded_failure_modes

        # Calculate metrics based on ground truth and prediction
        if 'ground_truth' in fm_data.columns and 'prediction' in fm_data.columns:
            y_true = (fm_data['ground_truth'] == 'hallucination').astype(int)
            y_pred = (fm_data['prediction'] == 'hallucination').astype(int)

            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()

            # For grounded failure modes, use TNR (True Negative Rate) as main metric
            # For hallucination failure modes, use standard precision/recall/F1
            if is_grounded_mode:
                # These cases should all be classified as "grounded"
                # TNR = TN / (TN + FP) - how well we correctly identify grounded content
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tnr  # For display consistency, show TNR
                recall = tnr
                f1 = tnr  # Use TNR as the "F1-equivalent" for grounded modes
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
            is_grounded_mode = False

        failure_mode_stats.append({
            'Failure Mode': fm.replace('_', ' ').title(),
            'Cases': total,
            'Correct': int(correct),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'is_grounded_mode': is_grounded_mode
        })

    if failure_mode_stats:
        fm_df = pd.DataFrame(failure_mode_stats)

        # Visual cards for each failure mode
        cols = st.columns(3)
        for i, row in fm_df.iterrows():
            with cols[i % 3]:
                # Determine status color based on accuracy (universal metric)
                acc_val = row['Accuracy']
                if acc_val >= 0.75:
                    status_class = "status-good"
                    status_color = COLORS['good']
                elif acc_val >= 0.60:
                    status_class = "status-warning"
                    status_color = COLORS['amber']
                else:
                    status_class = "status-poor"
                    status_color = COLORS['poor']

                # Use appropriate metric label based on failure mode type
                is_grounded = row.get('is_grounded_mode', False)
                metric_value = row['F1']  # F1 holds TNR for grounded modes

                # Beginner-friendly descriptions
                if is_grounded:
                    metric_label = "Accuracy"
                    mode_note = "Should say 'safe' for these"
                else:
                    metric_label = "Catch rate"
                    mode_note = "Should catch these errors"

                st.markdown(f"""
                <div class="metric-card {status_class}" style="padding: 1rem; margin-bottom: 0.75rem;">
                    <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.25rem; font-size: 0.9rem;">
                        {row['Failure Mode']}
                    </div>
                    <div style="font-size: 0.7rem; color: {COLORS['medium_gray']}; margin-bottom: 0.5rem;">
                        {mode_note}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 1.5rem; font-weight: 600; color: {status_color};">{metric_value:.0%}</span>
                        <span style="font-size: 0.8rem; color: {COLORS['medium_gray']};">{metric_label}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: {COLORS['medium_gray']}; margin-top: 0.5rem;">
                        {row['Cases']} cases · {row['Correct']}/{row['Cases']} correct
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Summary table
        with st.expander("View detailed table"):
            display_df = fm_df.copy()
            # Rename columns based on mode type
            display_df['Type'] = display_df['is_grounded_mode'].apply(lambda x: 'Grounded' if x else 'Hallucination')
            display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.1%}")
            display_df['Score'] = display_df.apply(
                lambda row: f"{row['F1']:.1%} (TNR)" if row['is_grounded_mode'] else f"{row['F1']:.1%} (F1)",
                axis=1
            )
            # Select columns for display
            display_df = display_df[['Failure Mode', 'Type', 'Cases', 'Correct', 'Accuracy', 'Score']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Beginner-friendly explanation
        st.markdown(f"""
        <div style="background: #fefce8; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.85rem;">
            <strong>Understanding the scores:</strong>
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.25rem; color: {COLORS['charcoal']};">
                <li><strong>Factual Addition, Valid Inference, Verbatim Grounded</strong> - These are safe/correct responses.
                    The score shows how often the model correctly says "this is fine" (avoiding false alarms).</li>
                <li><strong>Fabrication, Subtle Distortion, Fluent Hallucination, Partial Grounding</strong> - These contain errors.
                    The score shows how often the model catches these problems.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # SLICE BY DIFFICULTY
    # ==========================================
    render_section_header("By Difficulty")

    # Beginner-friendly explanation
    st.markdown(f"""
    <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 3px solid {COLORS['teal']};">
        <div style="font-size: 0.85rem; color: {COLORS['charcoal']}; line-height: 1.6;">
            <strong>What this means:</strong> Test cases are labeled by how tricky they are for humans to classify.
            Easy cases are obvious; hard cases require careful reading. If the model struggles on easy cases,
            something is wrong. If it struggles on hard cases, that's expected.
        </div>
    </div>
    """, unsafe_allow_html=True)

    difficulty_order = ['easy', 'medium', 'hard']
    difficulty_stats = []

    for diff in difficulty_order:
        diff_data = run_data[run_data['difficulty'] == diff]
        total = len(diff_data)
        if total == 0:
            continue

        correct = diff_data['correct'].sum() if 'correct' in diff_data.columns else 0
        accuracy = correct / total if total > 0 else 0

        # Avg confidence
        avg_conf = diff_data['confidence'].mean() if 'confidence' in diff_data.columns else 0

        difficulty_stats.append({
            'Difficulty': diff.title(),
            'Cases': total,
            'Correct': int(correct),
            'Accuracy': accuracy,
            'Avg Confidence': avg_conf
        })

    if difficulty_stats:
        diff_df = pd.DataFrame(difficulty_stats)

        cols = st.columns(3)
        for i, row in diff_df.iterrows():
            with cols[i]:
                acc_val = row['Accuracy']
                if acc_val >= 0.80:
                    status_color = COLORS['good']
                elif acc_val >= 0.65:
                    status_color = COLORS['amber']
                else:
                    status_color = COLORS['poor']

                # Icon based on difficulty
                diff_icon = "🟢" if row['Difficulty'] == 'Easy' else ("🟡" if row['Difficulty'] == 'Medium' else "🔴")

                st.markdown(f"""
                <div class="metric-card" style="padding: 1rem; text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{diff_icon}</div>
                    <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">
                        {row['Difficulty']}
                    </div>
                    <div style="font-size: 1.75rem; font-weight: 600; color: {status_color};">
                        {row['Accuracy']:.0%}
                    </div>
                    <div style="font-size: 0.75rem; color: {COLORS['medium_gray']};">
                        {row['Correct']}/{row['Cases']} correct
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # SLICE BY LABEL (Ground Truth)
    # ==========================================
    render_section_header("By Ground Truth Label")

    # Beginner-friendly explanation
    st.markdown(f"""
    <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 3px solid {COLORS['teal']};">
        <div style="font-size: 0.85rem; color: {COLORS['charcoal']}; line-height: 1.6;">
            <strong>What this shows:</strong> How well the model performs on each type of content:
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.25rem;">
                <li><strong>Grounded</strong> = Safe, accurate content. Does the model correctly say "this is fine"?</li>
                <li><strong>Hallucination</strong> = Content with errors. Does the model catch these problems?</li>
            </ul>
            If one score is much lower than the other, the model is biased toward one type of prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'ground_truth' in run_data.columns:
        label_stats = []

        for label in ['grounded', 'hallucination']:
            label_data = run_data[run_data['ground_truth'] == label]
            total = len(label_data)
            if total == 0:
                continue

            correct = label_data['correct'].sum() if 'correct' in label_data.columns else 0
            accuracy = correct / total if total > 0 else 0
            avg_conf = label_data['confidence'].mean() if 'confidence' in label_data.columns else 0

            label_stats.append({
                'Label': label.title(),
                'Cases': total,
                'Correct': int(correct),
                'Accuracy': accuracy,
                'Avg Confidence': avg_conf
            })

        if label_stats:
            cols = st.columns(2)
            for i, row in enumerate(label_stats):
                with cols[i]:
                    acc_val = row['Accuracy']
                    if acc_val >= 0.80:
                        status_color = COLORS['good']
                    elif acc_val >= 0.65:
                        status_color = COLORS['amber']
                    else:
                        status_color = COLORS['poor']

                    label_color = COLORS['good'] if row['Label'] == 'Grounded' else COLORS['poor']

                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1.25rem; border-left: 4px solid {label_color};">
                        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem; font-size: 1rem;">
                            {row['Label']} Cases
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: {COLORS['charcoal']};">Accuracy</span>
                            <span style="font-weight: 600; color: {status_color};">{row['Accuracy']:.1%}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: {COLORS['charcoal']};">Avg Confidence</span>
                            <span style="font-weight: 500;">{row['Avg Confidence']:.1%}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: {COLORS['medium_gray']}; margin-top: 0.5rem;">
                            {row['Correct']}/{row['Cases']} correctly classified
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================
    # KEY INSIGHTS
    # ==========================================
    st.markdown("---")
    render_section_header("Key Insights", "What the numbers are telling you")

    insights = []

    # Find worst failure mode
    if failure_mode_stats:
        worst_fm = min(failure_mode_stats, key=lambda x: x['F1'])
        best_fm = max(failure_mode_stats, key=lambda x: x['F1'])

        if worst_fm['F1'] < 0.70:
            insights.append({
                'icon': '⚠️',
                'title': f"Struggles with {worst_fm['Failure Mode']}",
                'detail': f"Only {worst_fm['F1']:.0%} accuracy. This is an area to improve — consider adding more examples of this type to your prompt."
            })

        if best_fm['F1'] > 0.85:
            insights.append({
                'icon': '✓',
                'title': f"Good at {best_fm['Failure Mode']}",
                'detail': f"{best_fm['F1']:.0%} accuracy shows the model handles this type well."
            })

        gap = best_fm['F1'] - worst_fm['F1']
        if gap > 0.20:
            insights.append({
                'icon': '📊',
                'title': f"Uneven performance ({gap:.0%} gap)",
                'detail': f"Big difference between best ({best_fm['Failure Mode']}) and worst ({worst_fm['Failure Mode']}) areas. Focus on improving weak spots."
            })

    # Difficulty insights
    if difficulty_stats:
        easy_acc = next((d['Accuracy'] for d in difficulty_stats if d['Difficulty'] == 'Easy'), None)
        hard_acc = next((d['Accuracy'] for d in difficulty_stats if d['Difficulty'] == 'Hard'), None)

        if easy_acc and hard_acc:
            drop = easy_acc - hard_acc
            if drop > 0.15:
                insights.append({
                    'icon': '📉',
                    'title': f"Drops {drop:.0%} on hard cases",
                    'detail': "Performance falls significantly on tricky cases. This is somewhat expected, but large drops may indicate the model needs better reasoning guidance."
                })
            elif drop < 0.05:
                insights.append({
                    'icon': '✓',
                    'title': "Consistent across difficulty levels",
                    'detail': "The model performs similarly on easy and hard cases — good sign of robust performance."
                })

    if not insights:
        insights.append({
            'icon': '✓',
            'title': "Balanced performance",
            'detail': "No major issues detected. Performance is relatively consistent across all categories."
        })

    for insight in insights:
        st.markdown(f"""
        <div style="padding: 1rem; background: {COLORS['light_gray']}50; border-radius: 8px; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <span style="font-size: 1.25rem;">{insight['icon']}</span>
                <div>
                    <div style="font-weight: 600; color: {COLORS['navy']}; margin-bottom: 0.25rem;">{insight['title']}</div>
                    <div style="font-size: 0.85rem; color: {COLORS['charcoal']}; line-height: 1.5;">{insight['detail']}</div>
                </div>
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

                    # Summary cards - top row with cost
                    cost_usd = getattr(summary, 'total_cost_usd', 0.0)
                    total_tokens = getattr(summary, 'total_tokens', 0)

                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 1rem; padding: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: {COLORS['medium_gray']}; font-size: 0.85rem;">Run ID:</span>
                                <span style="font-family: monospace; color: {COLORS['navy']}; margin-left: 0.5rem;">{summary.run_id}</span>
                            </div>
                            <div style="display: flex; gap: 1.5rem;">
                                <div>
                                    <span style="color: {COLORS['medium_gray']}; font-size: 0.85rem;">Tokens:</span>
                                    <span style="color: {COLORS['charcoal']}; font-weight: 500; margin-left: 0.25rem;">{total_tokens:,}</span>
                                </div>
                                <div>
                                    <span style="color: {COLORS['medium_gray']}; font-size: 0.85rem;">Cost:</span>
                                    <span style="color: {COLORS['teal']}; font-weight: 600; margin-left: 0.25rem;">${cost_usd:.4f}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card status-good">
                            <span class="metric-label">Passed</span>
                            <div class="metric-value" style="color: {COLORS['good']};">{summary.scenarios_passed}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
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

                    st.success("✓ Results saved to database!")
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 1rem; padding: 1rem;">
                        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Next Steps</div>
                        <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.8;">
                            • Go to <strong>Metrics Overview</strong> to see detailed results<br>
                            • Go to <strong>Run History</strong> to browse all runs<br>
                            • Go to <strong>Compare Runs</strong> to compare with previous evaluations
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Evaluation failed. Please check your API key and try again.")

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

    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <a href="https://github.com/rasiulyte/evaluation-system/blob/main/docs/METRICS.md" target="_blank"
           style="color: {COLORS['teal']}; text-decoration: none; font-size: 0.9rem;">
            📄 View full documentation on GitHub →
        </a>
    </div>
    """, unsafe_allow_html=True)

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

        **✅ Works well when:**
        - **Spam filtering**: If every email marked as spam is actually spam, users trust the filter and don't lose important messages
        - **Content moderation**: When flagging inappropriate content, false alarms frustrate good users
        - **Chatbot warnings**: When showing "I'm not sure about this answer," users need to trust those warnings

        **❌ Doesn't work well when:**
        - **Medical screening alone**: High precision but low recall means you might miss real diseases - dangerous!
        - **Security alerts**: If you only flag the most obvious attacks (high precision), sophisticated ones slip through
        - **Legal document review**: Missing a problematic clause is worse than flagging too many for human review

        **⚠️ Risks:**
        - Can be artificially high if the model is too conservative (flags almost nothing)
        - A model that only flags the most obvious cases will have high precision but miss subtle hallucinations
        - Doesn't tell you how many hallucinations you're missing — always pair with Recall
        """)

    with st.expander("**Recall** — Catch Rate", expanded=True):
        st.markdown(f"""
        **Formula:** TP / (TP + FN)

        **Question answered:** Of all actual hallucinations, how many did we catch?

        **Intuition:** High recall means few hallucinations slip through. Critical for safety-sensitive
        applications where missing a hallucination could cause real harm (medical, legal, financial).

        **Target:** ≥ 0.75 for most applications, higher for safety-critical

        **✅ Works well when:**
        - **Medical diagnosis support**: Missing a disease is worse than ordering extra tests - catch everything
        - **Fraud detection**: Missing actual fraud costs money - better to review extra transactions
        - **Safety systems**: In self-driving cars, missing a pedestrian is catastrophic - flag all possibilities

        **❌ Doesn't work well when:**
        - **Email spam filtering alone**: Flagging everything as spam achieves 100% recall but your inbox is empty
        - **News fact-checking**: If every statement gets flagged for review, editors can't keep up
        - **Customer support**: Routing every query to a human (100% recall) defeats the purpose of automation

        **⚠️ Risks:**
        - Can be gamed by flagging everything as hallucination (100% recall but useless)
        - High recall with low precision means lots of false alarms — users will ignore warnings
        - Optimizing only for recall can make the system overly aggressive
        """)

    with st.expander("**F1 Score** — Balanced Performance", expanded=True):
        st.markdown(f"""
        **Formula:** 2 × (Precision × Recall) / (Precision + Recall)

        **Question answered:** How well does the model balance catching hallucinations vs. avoiding false alarms?

        **Intuition:** The harmonic mean punishes extreme imbalances. You can't game F1 by optimizing
        only precision or only recall. It's the primary metric for overall detection quality.

        **Target:** ≥ 0.75 indicates production-ready performance

        **✅ Works well when:**
        - **Comparing model versions**: "Which prompt detects hallucinations best overall?" - F1 gives a fair answer
        - **A/B testing**: Choosing between two approaches where neither error type is much worse than the other
        - **General benchmarking**: Standard way to report detection quality that's hard to game

        **❌ Doesn't work well when:**
        - **Medical AI**: Missing a tumor (FN) is much worse than a false alarm (FP) - prioritize recall instead
        - **Content moderation with legal risk**: Letting harmful content through is worse than over-blocking
        - **Very imbalanced data**: If 99% of content is safe, F1 may look good while missing most real issues

        **⚠️ Risks:**
        - Assumes precision and recall are equally important — they may not be in your use case
        - Doesn't account for class imbalance (if 95% of data is grounded, F1 can be misleading)
        - Two models with same F1 can have very different precision/recall trade-offs
        - For safety-critical apps, you may need to prioritize recall over F1
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

        **✅ Works well when:**
        - **Imbalanced datasets**: If 90% of your data is "safe," accuracy looks good but Kappa reveals the truth
        - **Comparing human labelers**: "Do two annotators agree beyond what you'd expect by chance?"
        - **Evaluating model improvement**: A Kappa jump from 0.4 to 0.6 is meaningful progress

        **❌ Doesn't work well when:**
        - **Very small test sets**: With only 20 examples, Kappa can swing wildly
        - **Rare events**: If hallucinations are extremely rare, even good detection gives low Kappa
        - **Comparing across different datasets**: Kappa depends on class distribution, so cross-dataset comparison is tricky

        **⚠️ Risks:**
        - Can be negative if model performs worse than random chance
        - Sensitive to prevalence — same performance looks different with different class distributions
        - Low Kappa doesn't always mean bad performance — it depends on baseline difficulty
        - Can be unstable with small sample sizes
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

        **✅ Works well when:**
        - **Small test sets**: With only 30-50 examples, Tau is more reliable than Pearson
        - **Many tied values**: If your model outputs "70%" confidence a lot, Tau handles ties gracefully
        - **Quick ranking check**: "Is the model's ordering of cases roughly correct?"

        **❌ Doesn't work well when:**
        - **Absolute calibration matters**: Tau = 0.8 doesn't mean 80% confidence = 80% accuracy
        - **Comparing to Pearson/Spearman**: Tau values are naturally lower - don't compare directly
        - **Binary confidence**: If model only says "confident" or "not confident", there's no ranking to measure

        **⚠️ Risks:**
        - Values are inherently lower than Spearman/Pearson — don't compare directly
        - Can be misleading if confidence scores cluster at extremes (all 0.9+ or all 0.1-)
        - Requires variance in both confidence and correctness — fails with constant predictions
        - Doesn't tell you if confidence values are calibrated, only if rankings are correct
        """)

    with st.expander("**Pearson Correlation** — Linear Relationship"):
        st.markdown(f"""
        **Question answered:** Is there a proportional relationship between confidence and correctness?

        **Intuition:** If confidence of 0.8 means 80% correct, that's perfect linear calibration.
        Sensitive to outliers.

        **Target:** ≥ 0.70

        **✅ Works well when:**
        - **Checking calibration**: "When the model says 80% confident, is it actually right 80% of the time?"
        - **Weather forecasting style**: Predictions that should map directly to probabilities
        - **Betting/ranking systems**: Where confidence should translate proportionally to outcomes

        **❌ Doesn't work well when:**
        - **Outliers exist**: One very wrong high-confidence prediction can tank your Pearson score
        - **Non-linear relationship**: Model might be well-calibrated but in a curved way (use Spearman)
        - **Clustered confidence**: If model only says 30%, 50%, or 90%, there's not enough spread to measure

        **⚠️ Risks:**
        - Assumes linear relationship — may miss valid non-linear calibration
        - Very sensitive to outliers — a few extreme values can distort results
        - Can be undefined if all predictions have same confidence (zero variance)
        - High Pearson doesn't mean well-calibrated — could have consistent bias
        """)

    with st.expander("**Spearman Correlation** — Rank-Order Relationship"):
        st.markdown(f"""
        **Question answered:** Do higher confidence predictions tend to be more correct?

        **Intuition:** Like Pearson but uses ranks instead of raw values. More robust to outliers
        and doesn't assume linearity.

        **Target:** ≥ 0.60

        **✅ Works well when:**
        - **Ranking is what matters**: "Show me the cases the model is most sure about" - Spearman validates this
        - **Outliers in your data**: More robust than Pearson when you have extreme values
        - **Non-linear but consistent**: Model's confidence grows with correctness, just not proportionally

        **❌ Doesn't work well when:**
        - **Absolute confidence matters**: Spearman = 0.9 doesn't mean confidence values are accurate
        - **You need exact calibration**: For "80% confident = 80% correct", use Pearson
        - **Many tied ranks**: If half your predictions have the same confidence, ranking breaks down

        **⚠️ Risks:**
        - Only measures monotonic relationships — doesn't require linear calibration
        - Can be high even if absolute confidence values are wrong (just needs correct ordering)
        - Sensitive to ties — many identical confidence values reduce reliability
        - Doesn't tell you if 80% confidence actually means 80% correct
        """)

    st.markdown("---")

    # Additional Classification Metrics
    render_section_header("Additional Classification Metrics")

    with st.expander("**TNR (True Negative Rate / Specificity)** — Protecting Good Content"):
        st.markdown(f"""
        **Formula:** TN / (TN + FP)

        **Question answered:** Of all grounded content, how much did we correctly accept?

        **Intuition:** High TNR means good content flows through freely. Low TNR means you're
        blocking legitimate content, frustrating users.

        **Target:** ≥ 0.65

        **✅ Works well when:**
        - **User experience matters**: Low TNR means good AI responses get flagged, annoying users
        - **Content publishing**: False alarms delay legitimate articles - you need high TNR
        - **Customer support bots**: Flagging correct answers makes the bot seem unreliable

        **❌ Doesn't work well when:**
        - **Safety is primary concern**: TNR = 100% by never flagging anything - but you miss all hallucinations
        - **Rare hallucinations**: If only 5% of content has issues, TNR naturally looks good
        - **Used alone**: Always pair with Recall to see the full picture

        **⚠️ Risks:**
        - Can be artificially high if the system rarely flags anything
        - Doesn't tell you how many hallucinations you're catching
        - Easy to achieve high TNR by being permissive — always pair with Recall
        - In imbalanced datasets, TNR can be misleading about overall performance
        """)

    with st.expander("**Accuracy** — Overall Correctness"):
        st.markdown(f"""
        **Formula:** (TP + TN) / (TP + TN + FP + FN)

        **Question answered:** What percentage of all predictions were correct?

        **Intuition:** Simple and intuitive, but can be deeply misleading.

        **Target:** ≥ 0.75

        **✅ Works well when:**
        - **Balanced datasets**: 50% hallucinations, 50% grounded - accuracy gives a fair picture
        - **Quick sanity check**: "Is this model doing something useful?" - accuracy below 50% is a red flag
        - **Explaining to non-technical stakeholders**: "The model is right 80% of the time" is easy to understand

        **❌ Doesn't work well when:**
        - **Imbalanced data (the classic trap!)**: If 95% of content is safe, predicting "safe" always gives 95% accuracy while catching zero hallucinations!
        - **Rare event detection**: Fraud detection, disease screening, hallucination detection - accuracy hides failures
        - **Comparing models**: Two models with same accuracy can have wildly different precision/recall

        **⚠️ Risks:**
        - **MAJOR RISK:** Extremely misleading with imbalanced data
        - If 95% of content is grounded, predicting "grounded" always gives 95% accuracy!
        - Hides poor performance on the minority class (usually hallucinations)
        - Should almost never be your primary metric — use F1 or Kappa instead
        - Only meaningful when classes are roughly balanced
        """)

    st.markdown("---")

    # Calibration Metrics
    render_section_header("Calibration Metrics")

    st.markdown("""
    These metrics measure how **accurate** the confidence scores are, not just their ranking.
    """)

    with st.expander("**Bias** — Systematic Over/Under-Prediction"):
        st.markdown(f"""
        **Formula:** Mean(Predicted) - Mean(Actual)

        **Question answered:** Does the model systematically over-flag or under-flag hallucinations?

        **Intuition:** Positive bias = too aggressive (flags too much). Negative bias = too lenient (misses too much).

        **Target:** |bias| ≤ 0.15

        **✅ Works well when:**
        - **Detecting systematic issues**: "This model flags 30% of content but only 15% is actually wrong" - clear bias problem
        - **Prompt tuning**: Bias helps you see if a new prompt made the model too aggressive or too lenient
        - **Monitoring drift**: If bias increases over time, something in your data or model has changed

        **❌ Doesn't work well when:**
        - **Errors cancel out**: Model misses some hallucinations AND falsely flags some good content - bias = 0 but model is wrong!
        - **Small samples**: Bias swings a lot with few examples
        - **Mixed difficulty**: Model might be biased differently on easy vs hard cases (need slice analysis)

        **⚠️ Risks:**
        - Zero bias doesn't mean good predictions — errors could cancel out
        - Can hide large errors if they're symmetric around zero
        - Sensitive to class distribution — recalculate when data changes
        - Doesn't tell you about individual prediction quality
        """)

    with st.expander("**MAE (Mean Absolute Error)** — Average Confidence Error"):
        st.markdown(f"""
        **Formula:** Mean(|Confidence - Actual|)

        **Question answered:** On average, how far off are the confidence scores?

        **Intuition:** Lower is better. MAE of 0.2 means confidence is typically off by 20%.

        **Target:** < 0.20

        **✅ Works well when:**
        - **Typical error matters**: "On average, how wrong is the model?" - useful for everyday performance
        - **Comparing calibration methods**: Which approach gives more accurate confidence scores overall?
        - **Budget planning**: MAE tells you how much manual review to expect on average

        **❌ Doesn't work well when:**
        - **Big mistakes are catastrophic**: A model with mostly small errors but one 90% confidence wrong answer still has low MAE
        - **You need worst-case analysis**: MAE hides outliers - use RMSE to catch them
        - **Decision thresholds**: Low MAE doesn't mean confidence threshold decisions work well

        **⚠️ Risks:**
        - Treats all errors equally — a 0.1 error and a 0.9 error average to 0.5
        - Doesn't penalize large errors more than small ones
        - Can be low even with a few catastrophically wrong predictions
        - Sensitive to confidence score distribution
        """)

    with st.expander("**RMSE (Root Mean Squared Error)** — Penalizes Large Errors"):
        st.markdown(f"""
        **Formula:** √(Mean((Confidence - Actual)²))

        **Question answered:** How bad are the worst confidence errors?

        **Intuition:** Like MAE but penalizes large errors more heavily. A few big mistakes
        hurt RMSE more than many small ones.

        **Target:** < 0.25

        **✅ Works well when:**
        - **Big mistakes are costly**: In finance, one confident wrong prediction can be worse than many small errors
        - **Catching overconfident failures**: Model says 95% confident but is wrong - RMSE catches this
        - **Comparing with MAE**: If RMSE >> MAE, you have an outlier problem to fix

        **❌ Doesn't work well when:**
        - **Outliers are expected**: One genuinely weird case can tank RMSE even if model is otherwise good
        - **You want average performance**: RMSE overweights rare large errors - use MAE instead
        - **Comparing models with different data**: A model tested on harder cases will have higher RMSE even if better

        **⚠️ Risks:**
        - More sensitive to outliers than MAE — one bad prediction can dominate
        - Harder to interpret than MAE (squared then rooted)
        - Can improve by removing outliers rather than fixing them
        - Should be compared alongside MAE — large gap suggests outlier issues
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
# PAGE: HOME
# ============================================

def render_home_page():
    """Getting Started page explaining the dashboard."""

    render_page_header(
        "Getting Started",
        "A sandbox for learning LLM-as-Judge evaluation"
    )

    st.markdown(f"""
    <div class="metric-card" style="padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.1rem; color: {COLORS['navy']}; margin-bottom: 1rem; font-weight: 500;">
            What is LLM-as-Judge?
        </div>
        <div style="color: {COLORS['charcoal']}; line-height: 1.7;">
            <strong>LLM-as-Judge</strong> (also called "autograder") is a technique where one LLM evaluates
            the output of another LLM. Instead of humans manually reviewing every AI response,
            we use a "judge" LLM to automatically classify responses as correct or incorrect.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Use case explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem;">In This System</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                We use <strong>GPT-4o-mini</strong> as the judge to detect <strong>hallucinations</strong> —
                cases where an AI makes up information or contradicts the source material.
                The judge reads a context + response pair and decides: is this grounded or hallucinated?
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem;">Why Use LLM-as-Judge?</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                <strong>Scale:</strong> Evaluate thousands of responses automatically<br>
                <strong>Speed:</strong> Get results in seconds, not days<br>
                <strong>Consistency:</strong> Same criteria applied every time<br>
                <strong>Cost:</strong> Cheaper than human annotation at scale
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation workflow (at top for visibility)
    render_section_header("How Evaluation Works")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['navy']}08, {COLORS['teal']}08);
                padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <p style="margin: 0 0 1rem 0; color: {COLORS['charcoal']};">
            When you click <strong>Run Evaluation</strong>, here's exactly what happens behind the scenes:
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Workflow steps
    workflow_steps = [
        {
            "num": "1",
            "title": "Load Test Cases",
            "desc": "The system loads test cases from <code>data/test_cases/</code>. Each test case has a <strong>context</strong> (source material), a <strong>response</strong> (text to evaluate), and an expected <strong>label</strong> (grounded or hallucination).",
            "detail": "A random sample is selected based on the configured sample size (default: 20 cases)."
        },
        {
            "num": "2",
            "title": "Load Prompt Template",
            "desc": "The selected prompt template (e.g., <code>v6_calibrated_confidence</code>) is loaded from <code>prompts/</code>. This template tells the LLM how to analyze the response.",
            "detail": "The prompt includes instructions for classification and confidence calibration guidelines."
        },
        {
            "num": "3",
            "title": "Format & Send to LLM",
            "desc": "For each test case, the context and response are inserted into the prompt template. The formatted prompt is sent to the LLM (e.g., GPT-4o-mini).",
            "detail": "Example: \"Context: {context}\\n\\nResponse to analyze: {response}\\n\\nIs this grounded or hallucinated?\""
        },
        {
            "num": "4",
            "title": "Parse LLM Response",
            "desc": "The LLM returns a JSON response with <code>classification</code>, <code>confidence</code>, and <code>reasoning</code>. The system parses this to extract the prediction.",
            "detail": "Example output: {\"classification\": \"hallucinated\", \"confidence\": 0.85, \"reasoning\": \"The response adds information not in context...\"}"
        },
        {
            "num": "5",
            "title": "Compare to Ground Truth",
            "desc": "Each LLM prediction is compared against the expected label from the test case. A prediction is <span style='color: {COLORS['good']};'>correct</span> if it matches, <span style='color: {COLORS['poor']};'>wrong</span> if it doesn't.",
            "detail": "This comparison creates the raw data for calculating metrics."
        },
        {
            "num": "6",
            "title": "Calculate Metrics",
            "desc": "Using all predictions vs ground truth labels, the system calculates metrics: F1, Precision, Recall, TNR, Accuracy, Cohen's Kappa, and correlation metrics (Spearman, etc.).",
            "detail": "Calibration metrics (Bias, MAE, RMSE) are calculated from the confidence scores."
        },
        {
            "num": "7",
            "title": "Save Results",
            "desc": "Everything is saved: individual test results (with LLM reasoning) go to <code>data/daily_runs/</code> and aggregated metrics go to the database for the dashboard.",
            "detail": "Results are timestamped so you can track changes over time and compare runs."
        }
    ]

    for i, step in enumerate(workflow_steps):
        bg_color = f"{COLORS['teal']}08" if i % 2 == 0 else f"{COLORS['navy']}05"
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; padding: 1rem; background: {bg_color}; border-radius: 8px; border-left: 4px solid {COLORS['teal']};">
            <div style="background: {COLORS['teal']}; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">
                {step['num']}
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.25rem;">{step['title']}</div>
                <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.5;">{step['desc']}</div>
                <div style="color: {COLORS['medium_gray']}; font-size: 0.8rem; margin-top: 0.5rem; font-style: italic;">{step['detail']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Visual diagram
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 1.5rem; padding: 1.5rem; text-align: center;">
        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 1rem;">Data Flow Summary</div>
        <div style="font-family: monospace; font-size: 0.85rem; color: {COLORS['charcoal']}; line-height: 2;">
            <span style="background: {COLORS['light_gray']}; padding: 0.25rem 0.5rem; border-radius: 4px;">Test Cases</span>
            <span style="color: {COLORS['teal']}; margin: 0 0.5rem;">→</span>
            <span style="background: {COLORS['light_gray']}; padding: 0.25rem 0.5rem; border-radius: 4px;">+ Prompt</span>
            <span style="color: {COLORS['teal']}; margin: 0 0.5rem;">→</span>
            <span style="background: {COLORS['teal']}20; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid {COLORS['teal']};">LLM</span>
            <span style="color: {COLORS['teal']}; margin: 0 0.5rem;">→</span>
            <span style="background: {COLORS['light_gray']}; padding: 0.25rem 0.5rem; border-radius: 4px;">Predictions</span>
            <span style="color: {COLORS['teal']}; margin: 0 0.5rem;">→</span>
            <span style="background: {COLORS['light_gray']}; padding: 0.25rem 0.5rem; border-radius: 4px;">vs Ground Truth</span>
            <span style="color: {COLORS['teal']}; margin: 0 0.5rem;">→</span>
            <span style="background: {COLORS['good']}20; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid {COLORS['good']};">Metrics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How to use section
    render_section_header("How to Use This Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">1. Explore Metrics</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Start with <strong>Metrics Overview</strong> to see evaluation results with interpretation guides.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="height: 100%; margin-top: 1rem;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">2. Compare Runs</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Use <strong>Compare Runs</strong> and <strong>Trends</strong> to analyze differences between evaluations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">3. Browse Test Cases</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Visit <strong>Test Cases</strong> to see all test data — context, response, and expected labels.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="height: 100%; margin-top: 1rem;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">4. Learn Failure Modes</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Visit <strong>Failure Modes</strong> to understand the different ways AI can hallucinate.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">5. Explore Prompt Lab</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Learn hillclimbing and see how prompts affect results in <strong>Prompt Lab</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="height: 100%; margin-top: 1rem;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">6. Understand the Theory</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                Read <strong>Understanding Metrics</strong> or the <a href="https://github.com/rasiulyte/evaluation-system/blob/main/docs/METRICS.md" target="_blank" style="color: {COLORS['teal']};">Metrics Guide</a> on GitHub.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Key concepts section
    render_section_header("Key Concepts")

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 1rem;">
        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Hallucination Detection</div>
        <div style="color: {COLORS['charcoal']}; font-size: 0.9rem;">
            AI models sometimes generate content that sounds plausible but is factually incorrect or
            unsupported by the given context. Detecting these "hallucinations" is critical for building
            trustworthy AI systems.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem;">Classification</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.85rem;">
                F1, Precision, Recall — How well does the system identify hallucinations vs grounded content?
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem;">Correlation</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.85rem;">
                Spearman, Kendall's Tau — Does the model's confidence actually predict correctness?
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem;">Calibration</div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.85rem;">
                Bias, MAE, RMSE — Are the confidence scores well-calibrated and unbiased?
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick start checklist
    render_section_header("Quick Start Checklist")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['teal']}08, {COLORS['navy']}05);
                padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="font-size: 0.95rem; color: {COLORS['charcoal']}; margin-bottom: 1rem;">
            New here? Follow these steps to get started:
        </div>
    </div>
    """, unsafe_allow_html=True)

    checklist_items = [
        {"step": "1", "title": "Learn the basics", "desc": "Read <strong>Failure Modes</strong> to understand how AI hallucinations happen", "page": "Failure Modes"},
        {"step": "2", "title": "Explore test cases", "desc": "Browse <strong>Test Cases</strong> to see real examples of grounded vs hallucinated responses", "page": "Test Cases"},
        {"step": "3", "title": "Run an evaluation", "desc": "Go to <strong>Run Evaluation</strong> to test the LLM judge on sample data", "page": "Run Evaluation"},
        {"step": "4", "title": "Analyze results", "desc": "Check <strong>Metrics Overview</strong> to see how well the judge performed", "page": "Metrics Overview"},
    ]

    for item in checklist_items:
        col1, col2 = st.columns([0.92, 0.08])
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="background: {COLORS['teal']}; color: white; width: 24px; height: 24px;
                                border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                font-size: 0.8rem; font-weight: bold; flex-shrink: 0;">{item['step']}</div>
                    <div>
                        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.25rem;">{item['title']}</div>
                        <div style="color: {COLORS['charcoal']}; font-size: 0.85rem;">{item['desc']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Go →", key=f"checklist_{item['step']}", use_container_width=True):
                st.session_state.current_page = item['page']
                st.rerun()

    # Audio Overview (at bottom)
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    render_section_header("Audio Overview")

    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem; margin-bottom: 1rem;">
        <div style="color: {COLORS['charcoal']}; font-size: 0.95rem; line-height: 1.6;">
            🎧 Prefer listening? Here's a quick audio introduction to how AI judges catch hallucinations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    audio_path = Path(__file__).parent.parent / "assets" / "How_AI_Judges_Catch_Hallucinations.m4a"
    if audio_path.exists():
        st.audio(str(audio_path), format="audio/m4a")
    else:
        st.info("Audio overview not available.")

    # Video Tutorial
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    render_section_header("Video Tutorial")

    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem; margin-bottom: 1rem;">
        <div style="color: {COLORS['charcoal']}; font-size: 0.95rem; line-height: 1.6;">
            🎬 Watch a walkthrough explaining how to grade an AI using this evaluation system.
        </div>
    </div>
    """, unsafe_allow_html=True)

    video_path = Path(__file__).parent.parent / "assets" / "How_to_Grade_an_AI.mp4"
    if video_path.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.video(str(video_path))
    else:
        st.info("Video tutorial not available.")


# ============================================
# PAGE: TEST CASES
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_all_test_cases() -> list:
    """Load all test cases from the test_cases directory."""
    import json
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    test_cases_dir = project_root / "data" / "test_cases"

    all_cases = []
    if not test_cases_dir.exists():
        return all_cases

    for json_file in sorted(test_cases_dir.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                cases = json.load(f)
                for case in cases:
                    case['_source_file'] = json_file.name
                    all_cases.append(case)
        except Exception:
            continue

    return all_cases


def render_test_cases_page():
    """Page to browse all test cases."""

    render_page_header(
        "Test Cases",
        "Browse all test cases used for evaluation"
    )

    # Load all test cases
    all_cases = load_all_test_cases()

    if not all_cases:
        st.info("No test cases found in data/test_cases/")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        render_simple_metric("Total Cases", str(len(all_cases)))
    with col2:
        hallucination_count = len([c for c in all_cases if c.get('label') == 'hallucination'])
        render_simple_metric("Hallucinations", str(hallucination_count))
    with col3:
        grounded_count = len([c for c in all_cases if c.get('label') == 'grounded'])
        render_simple_metric("Grounded", str(grounded_count))

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        labels = sorted(set(c.get('label', 'unknown') for c in all_cases))
        selected_label = st.selectbox("Filter by Label", ["All"] + labels, key="tc_label_filter")

    with col2:
        failure_modes = sorted(set(c.get('failure_mode', '') for c in all_cases if c.get('failure_mode')))
        selected_fm = st.selectbox("Filter by Failure Mode", ["All"] + failure_modes, key="tc_fm_filter")

    # Apply filters
    filtered_cases = all_cases
    if selected_label != "All":
        filtered_cases = [c for c in filtered_cases if c.get('label') == selected_label]
    if selected_fm != "All":
        filtered_cases = [c for c in filtered_cases if c.get('failure_mode') == selected_fm]

    st.markdown(f"**Showing {len(filtered_cases)} test cases**")

    # Display test cases
    for case in filtered_cases:
        case_id = case.get('id', 'unknown')
        label = case.get('label', '—')
        failure_mode = case.get('failure_mode', '')

        # Color based on label
        if label == 'hallucination':
            label_color = COLORS['poor']
            status_class = "status-poor"
        elif label == 'grounded':
            label_color = COLORS['good']
            status_class = "status-good"
        else:
            label_color = COLORS['medium_gray']
            status_class = ""

        with st.expander(f"**{case_id}** — {label}" + (f" — {failure_mode}" if failure_mode else "")):
            # Metadata row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Label:** <span style='color: {label_color};'>{label}</span>", unsafe_allow_html=True)
            with col2:
                if failure_mode:
                    st.markdown(f"**Failure Mode:** `{failure_mode}`")
            with col3:
                st.markdown(f"**Source:** `{case.get('_source_file', '—')}`")

            st.markdown("---")

            # Context
            st.markdown("**Context** (source material):")
            context = case.get('context', '—')
            st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.9rem; margin-bottom: 1rem; white-space: pre-wrap;">{context}</div>""", unsafe_allow_html=True)

            # Response
            st.markdown("**Response** (text to evaluate for hallucinations):")
            response = case.get('response', '—')
            st.markdown(f"""<div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 6px; font-size: 0.9rem; border-left: 4px solid {label_color}; white-space: pre-wrap;">{response}</div>""", unsafe_allow_html=True)

            # Explanation if available
            if case.get('explanation'):
                st.markdown("**Explanation:**")
                st.markdown(f"_{case.get('explanation')}_")


# ============================================
# PAGE: LIMITATIONS
# ============================================

def render_limitations_page():
    """Page explaining what this evaluation system does NOT cover."""

    render_page_header(
        "Limitations",
        "What this evaluation system is and isn't"
    )

    # Main disclaimer
    st.markdown(f"""
    <div class="metric-card status-warning" style="padding: 1.5rem; margin-bottom: 2rem;">
        <div style="font-size: 1.1rem; color: {COLORS['navy']}; margin-bottom: 0.75rem; font-weight: 500;">
            This is a Learning Sandbox
        </div>
        <div style="color: {COLORS['charcoal']}; line-height: 1.7;">
            This project is designed to help you <strong>learn about LLM evaluation concepts</strong>.
            It demonstrates one specific approach to one specific problem. Real-world evaluation systems
            are far more comprehensive.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What this system IS
    render_section_header("What This System Does")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card status-good" style="padding: 1.25rem; height: 100%;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">
                ✓ Focuses on Hallucination Detection
            </div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                This system specifically evaluates whether an AI response is <strong>grounded</strong>
                (supported by the given context) or <strong>hallucinated</strong> (contains made-up information).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card status-good" style="padding: 1.25rem; height: 100%;">
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.75rem;">
                ✓ Uses LLM-as-Judge Approach
            </div>
            <div style="color: {COLORS['charcoal']}; font-size: 0.9rem; line-height: 1.6;">
                We use one LLM (the "judge") to evaluate the outputs of another LLM.
                This is just <strong>one of many evaluation methods</strong> — not the only or best approach.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # What this system is NOT
    render_section_header("What This System Does NOT Do")

    st.markdown(f"""
    <div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <div style="color: {COLORS['charcoal']}; font-size: 0.95rem;">
            LLM evaluation is multidimensional. This sandbox only scratches the surface.
            Here are important dimensions we <strong>do not</strong> evaluate:
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dimensions not covered
    dimensions_not_covered = [
        {
            "title": "Helpfulness & Relevance",
            "desc": "Does the response actually answer the user's question? Is it useful and on-topic?",
            "icon": "💡"
        },
        {
            "title": "Harmlessness & Safety",
            "desc": "Does the response avoid harmful content? Does it refuse inappropriate requests?",
            "icon": "🛡️"
        },
        {
            "title": "Honesty & Transparency",
            "desc": "Does the model express uncertainty appropriately? Does it admit when it doesn't know?",
            "icon": "🎯"
        },
        {
            "title": "Coherence & Fluency",
            "desc": "Is the text well-written? Is it grammatically correct and easy to understand?",
            "icon": "✍️"
        },
        {
            "title": "Reasoning & Logic",
            "desc": "Does the model follow logical steps? Are its conclusions sound?",
            "icon": "🧠"
        },
        {
            "title": "Instruction Following",
            "desc": "Does the model follow the format, length, and style requested by the user?",
            "icon": "📋"
        },
        {
            "title": "Creativity & Originality",
            "desc": "For creative tasks, is the output novel and interesting?",
            "icon": "🎨"
        },
        {
            "title": "Factual Accuracy (External)",
            "desc": "Is information correct based on world knowledge (not just the given context)?",
            "icon": "📚"
        },
    ]

    # Display in 2 columns
    col1, col2 = st.columns(2)

    for i, dim in enumerate(dimensions_not_covered):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                    <span style="font-size: 1.25rem;">{dim['icon']}</span>
                    <div>
                        <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.25rem;">
                            {dim['title']}
                        </div>
                        <div style="color: {COLORS['charcoal']}; font-size: 0.85rem; line-height: 1.5;">
                            {dim['desc']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Other evaluation methods
    render_section_header("Other Evaluation Methods We Don't Use")

    st.markdown(f"""
    <div style="background: {COLORS['light_gray']}; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <div style="color: {COLORS['charcoal']}; font-size: 0.95rem;">
            LLM-as-Judge is just one approach. Other methods include:
        </div>
    </div>
    """, unsafe_allow_html=True)

    methods = [
        {
            "title": "Human Evaluation",
            "desc": "Real humans rate responses — the gold standard but expensive and slow.",
        },
        {
            "title": "Automated Metrics",
            "desc": "BLEU, ROUGE, BERTScore — compare outputs to reference answers programmatically.",
        },
        {
            "title": "Benchmark Suites",
            "desc": "MMLU, HellaSwag, TruthfulQA — standardized tests with known correct answers.",
        },
        {
            "title": "Red Teaming",
            "desc": "Adversarial testing to find failure cases and safety issues.",
        },
        {
            "title": "A/B Testing",
            "desc": "Compare two models in production with real users.",
        },
        {
            "title": "Constitutional AI",
            "desc": "Self-critique where the model evaluates and improves its own outputs.",
        },
    ]

    col1, col2, col3 = st.columns(3)

    for i, method in enumerate(methods):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem; margin-bottom: 0.75rem; min-height: 120px;">
                <div style="font-weight: 500; color: {COLORS['teal']}; margin-bottom: 0.5rem; font-size: 0.9rem;">
                    {method['title']}
                </div>
                <div style="color: {COLORS['charcoal']}; font-size: 0.8rem; line-height: 1.5;">
                    {method['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Key takeaway
    render_section_header("Key Takeaway")

    st.markdown(f"""
    <div class="metric-card" style="padding: 1.5rem; border-left: 4px solid {COLORS['teal']};">
        <div style="color: {COLORS['charcoal']}; font-size: 1rem; line-height: 1.8;">
            <strong>A complete LLM evaluation strategy combines multiple methods and dimensions.</strong>
            <br><br>
            This sandbox teaches you the fundamentals of:
            <ul style="margin: 0.5rem 0 0 1.5rem; padding: 0;">
                <li>How to structure evaluation test cases</li>
                <li>How LLM-as-Judge works in practice</li>
                <li>How to interpret classification metrics</li>
                <li>How to iterate on prompts (hillclimbing)</li>
            </ul>
            <br>
            Use these concepts as building blocks for more comprehensive evaluation systems.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# PAGE: FAILURE MODES (Educational Guide)
# ============================================

def render_failure_modes_page():
    """Page explaining each failure mode in simple, educational terms."""

    render_page_header(
        "Failure Modes",
        "Understanding the different ways AI can hallucinate"
    )

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <a href="https://github.com/rasiulyte/evaluation-system/blob/main/docs/FAILURE_MODES.md" target="_blank"
           style="color: {COLORS['teal']}; text-decoration: none; font-size: 0.9rem;">
            📄 View full documentation on GitHub →
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['navy']}08, {COLORS['teal']}08);
                padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
        <p style="margin: 0; color: {COLORS['charcoal']}; line-height: 1.7;">
            <strong>What is a failure mode?</strong> It's a specific pattern or type of mistake that AI systems make
            when generating responses. Understanding these patterns helps you design better evaluation tests
            and build more reliable AI systems.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick reference table
    st.markdown("### Quick Reference")

    st.markdown(f"""
    | Failure Mode | In Simple Words | Risk Level |
    |-------------|-----------------|------------|
    | **Fabrication** | Making things up from scratch | High |
    | **Subtle Distortion** | Small changes that flip the meaning | High |
    | **Fluent Hallucination** | Confident lies that sound true | Very High |
    | **Partial Grounding** | Mixing truth with fiction | Medium-High |
    | **Factual Addition** | Adding info not in the source | Low-Medium |
    | **Valid Inference** | Logical conclusions from given facts | Low (usually OK) |
    | **Verbatim Grounded** | Repeating exactly what was given | None (baseline) |
    """)

    st.markdown("---")

    # Detailed explanations
    st.markdown("### Detailed Explanations")

    # FM2: Fabrication
    with st.expander("**Fabrication** — Making Things Up", expanded=True):
        st.markdown(f"""
        **What it is:**
        The AI invents information that has no basis in reality or directly contradicts the source material.
        This is the most obvious type of hallucination.

        **Simple Example:**
        - **Context:** "Paris is the capital of France."
        - **Hallucinated Response:** "Paris is the capital of Germany."

        **Why it happens:**
        - The model confuses similar concepts
        - Training data had conflicting information
        - The model is "filling in gaps" with plausible-sounding but wrong information

        **How to spot it:**
        - Claims that contradict basic facts
        - Specific dates, numbers, or names that seem oddly precise
        - Information that seems "too interesting" to be true

        **Risk Level:** <span style="color: {COLORS['poor']}; font-weight: bold;">HIGH</span>
        - Easy to detect with fact-checking
        - Can cause serious misinformation if not caught
        - Often involves verifiable facts that can be confirmed/denied
        """, unsafe_allow_html=True)

    # FM3: Subtle Distortion
    with st.expander("**Subtle Distortion** — Small Changes, Big Impact", expanded=False):
        st.markdown(f"""
        **What it is:**
        The AI makes small modifications to information that completely change its meaning.
        These are often harder to catch than outright fabrications because they look "almost right."

        **Simple Example:**
        - **Context:** "The study found that 25% of participants improved."
        - **Hallucinated Response:** "The study found that 75% of participants improved."

        **Common Types:**
        - **Number swaps:** 25% becomes 75%, or 500 becomes 50
        - **Direction reversals:** "increased" becomes "decreased"
        - **Qualifier changes:** "might help" becomes "will definitely help"
        - **Time shifts:** 2019 becomes 1992

        **Why it's dangerous:**
        The response still "sounds right" because most of it IS right. A reader might not notice
        that one crucial number or word has changed.

        **How to spot it:**
        - Compare numbers carefully against the source
        - Watch for direction words (up/down, more/less, better/worse)
        - Be suspicious of precise-sounding statistics

        **Risk Level:** <span style="color: {COLORS['poor']}; font-weight: bold;">HIGH</span>
        - Hard to detect because most of the text is accurate
        - Can completely reverse the meaning of findings
        - Particularly dangerous in medical, scientific, or financial contexts
        """, unsafe_allow_html=True)

    # FM6: Fluent Hallucination
    with st.expander("**Fluent Hallucination** — Confident Lies", expanded=False):
        st.markdown(f"""
        **What it is:**
        The AI produces well-written, authoritative-sounding text that is completely false.
        These hallucinations are dangerous because they SOUND credible and professional.

        **Simple Example:**
        - **Context:** "Machine learning is a subset of artificial intelligence."
        - **Hallucinated Response:** "Machine learning, pioneered by Alan Turing in the 1940s
          with his groundbreaking work on computational intelligence, revolutionized how we process information."

        (This sounds scholarly but is historically inaccurate — Turing worked on theory, not ML as we know it.)

        **Why it's the most dangerous:**
        - The writing quality creates false confidence
        - Includes specific names, dates, and technical terms that add apparent credibility
        - Readers often assume well-written = well-researched

        **Red flags to watch for:**
        - Overly specific attributions ("invented by X in year Y")
        - Claims about discoveries, inventions, or "breakthroughs"
        - Text that sounds like it came from a Wikipedia article you've never seen
        - Information that would be notable if true (and easily findable)

        **Risk Level:** <span style="color: {COLORS['poor']}; font-weight: bold;">VERY HIGH</span>
        - Hardest to detect because it sounds authoritative
        - Often passes human review if reviewers don't fact-check
        - Can spread as "knowledge" if published
        """, unsafe_allow_html=True)

    # FM7: Partial Grounding
    with st.expander("**Partial Grounding** — Mixing Truth with Fiction", expanded=False):
        st.markdown(f"""
        **What it is:**
        The response starts with accurate information from the source, then smoothly
        transitions into hallucinated content. The truth at the beginning builds trust
        that makes the lies easier to miss.

        **Simple Example:**
        - **Context:** "The study involved 100 participants from North America."
        - **Hallucinated Response:** "The study involved 100 participants from North America
          who were selected for their exceptional psychic abilities."

        The first part is true, the second part is completely made up.

        **Why it's tricky:**
        - The accurate opening creates credibility
        - Readers may stop fact-checking once they verify the first part
        - The transition from fact to fiction is often seamless

        **How to spot it:**
        - Read responses completely, not just the beginning
        - Be extra suspicious of "and also..." additions
        - Watch for claims that escalate in specificity or drama

        **Risk Level:** <span style="color: {COLORS['warning']}; font-weight: bold;">MEDIUM-HIGH</span>
        - Deceptive because the grounded portion builds trust
        - Often caught by reading to the end of responses
        - The hallucinated part is usually more dramatic/interesting than the source
        """, unsafe_allow_html=True)

    # FM1: Factual Addition
    with st.expander("**Factual Addition** — Adding Extra (But True) Information", expanded=False):
        st.markdown(f"""
        **What it is:**
        The AI adds information that wasn't in the source material, BUT the added
        information is actually true general knowledge. This is the "gray area" of hallucination.

        **Simple Example:**
        - **Context:** "Python is a programming language used for data science."
        - **Response:** "Python is a programming language used for data science,
          particularly with libraries like NumPy and Pandas."

        NumPy and Pandas ARE real Python libraries — but they weren't mentioned in the context.

        **Is this a problem?**
        It depends on your use case:
        - **Strict faithfulness required:** This IS a hallucination (info not in source)
        - **General accuracy acceptable:** This is fine (info is correct)

        **When it's OK:**
        - Casual Q&A where accuracy matters more than source-faithfulness
        - When adding well-established common knowledge

        **When it's NOT OK:**
        - Summarizing specific documents (should only include what's there)
        - Legal, medical, or compliance contexts
        - When traceability to source is important

        **Risk Level:** <span style="color: {COLORS['teal']}; font-weight: bold;">LOW-MEDIUM</span>
        - Added information is factually correct
        - May be appropriate depending on use case
        - Still technically unfaithful to the source
        """, unsafe_allow_html=True)

    # FM4: Valid Inference
    with st.expander("**Valid Inference** — Logical Conclusions (Usually OK)", expanded=False):
        st.markdown(f"""
        **What it is:**
        The AI draws a logical conclusion from the given information. The conclusion
        isn't stated explicitly in the source, but it FOLLOWS logically from what was said.

        **Simple Example:**
        - **Context:** "All mammals are vertebrates. Dogs are mammals."
        - **Response:** "Dogs are vertebrates."

        This conclusion isn't hallucination — it's basic logic (a syllogism).

        **Why this is different from hallucination:**
        - The conclusion is NECESSARILY TRUE given the premises
        - No new information is invented
        - The inference follows standard logical rules

        **When it's acceptable:**
        - Basic logical deductions (if A then B, A is true, therefore B)
        - Mathematical calculations from given numbers
        - Obvious implications ("it's raining" → "the ground is probably wet")

        **When to be careful:**
        - Inferences that require assumptions not stated
        - Probabilistic reasoning presented as certainty
        - Long chains of inference (each step adds uncertainty)

        **Risk Level:** <span style="color: {COLORS['good']}; font-weight: bold;">LOW</span>
        - Generally acceptable and expected
        - Part of what makes AI useful
        - Only problematic if the logic is flawed
        """, unsafe_allow_html=True)

    # FM5: Verbatim Grounded
    with st.expander("**Verbatim Grounded** — Exact Repetition (Baseline)", expanded=False):
        st.markdown(f"""
        **What it is:**
        The AI repeats or closely paraphrases exactly what was in the source material.
        This is the "gold standard" for faithfulness — no hallucination at all.

        **Simple Examples:**
        - **Context:** "Water boils at 100 degrees Celsius."
        - **Response:** "Water boils at 100 degrees Celsius." ✓

        - **Context:** "The Earth orbits the Sun approximately every 365 days."
        - **Response:** "The Earth takes about 365 days to orbit the Sun." ✓

        **Why we test for this:**
        - Establishes that the model CAN be faithful when it wants to
        - Provides baseline for comparison
        - Some use cases require strict verbatim accuracy

        **Variations that are still grounded:**
        - Word reordering ("A is B" → "B describes A")
        - Synonym substitution ("approximately" → "about")
        - Passive/active voice changes

        **Risk Level:** <span style="color: {COLORS['good']}; font-weight: bold;">NONE</span>
        - This is what you want
        - No new information added
        - Fully traceable to source
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Summary section
    st.markdown("### Designing Tests for Each Failure Mode")

    st.markdown(f"""
    When building test cases for your evaluation system, include examples from each failure mode:

    <div style="background: {COLORS['light_gray']}; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid {COLORS['medium_gray']};">
                <th style="text-align: left; padding: 0.5rem;">Failure Mode</th>
                <th style="text-align: left; padding: 0.5rem;">What to Test</th>
                <th style="text-align: left; padding: 0.5rem;">Expected Label</th>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>fabrication</code></td>
                <td style="padding: 0.5rem;">Completely false claims</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['poor']};">hallucination</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>subtle_distortion</code></td>
                <td style="padding: 0.5rem;">Small changes to numbers/direction</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['poor']};">hallucination</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>fluent_hallucination</code></td>
                <td style="padding: 0.5rem;">Well-written false claims</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['poor']};">hallucination</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>partial_grounding</code></td>
                <td style="padding: 0.5rem;">Mix of true + false</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['poor']};">hallucination</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>factual_addition</code></td>
                <td style="padding: 0.5rem;">True info not in source</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['teal']};">depends on strictness</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>valid_inference</code></td>
                <td style="padding: 0.5rem;">Logical deductions</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['good']};">grounded</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>verbatim_grounded</code></td>
                <td style="padding: 0.5rem;">Exact/close repetition</td>
                <td style="padding: 0.5rem;"><span style="color: {COLORS['good']};">grounded</span></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['teal']}10, {COLORS['navy']}05);
                padding: 1rem; border-radius: 8px; margin-top: 2rem;">
        <strong>Tip:</strong> A well-balanced test set includes cases from ALL failure modes.
        This ensures your hallucination detector can catch different types of mistakes,
        not just the obvious ones.
    </div>
    """, unsafe_allow_html=True)


# ============================================
# PAGE: PROMPT LAB (Hillclimbing)
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_all_prompts() -> dict:
    """Load all prompt templates from the prompts directory."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    prompts_dir = project_root / "prompts"

    prompts = {}
    if not prompts_dir.exists():
        return prompts

    for txt_file in sorted(prompts_dir.glob("*.txt")):
        try:
            with open(txt_file, "r") as f:
                prompts[txt_file.stem] = f.read()
        except Exception:
            continue

    return prompts


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_metrics_by_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """Get aggregated metrics grouped by prompt_id from test results."""
    try:
        # Get all test results
        test_results = db.get_test_results()
        if not test_results:
            return pd.DataFrame()

        results_df = pd.DataFrame(test_results)
        if 'prompt_id' not in results_df.columns:
            return pd.DataFrame()

        # Calculate metrics per prompt
        prompt_metrics = []
        for prompt_id in results_df['prompt_id'].unique():
            prompt_data = results_df[results_df['prompt_id'] == prompt_id]

            # Calculate basic metrics
            total = len(prompt_data)
            correct = prompt_data['correct'].sum() if 'correct' in prompt_data.columns else 0
            accuracy = correct / total if total > 0 else 0

            # Calculate average confidence
            avg_confidence = prompt_data['confidence'].mean() if 'confidence' in prompt_data.columns else 0

            prompt_metrics.append({
                'prompt_id': prompt_id,
                'total_cases': total,
                'correct': correct,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence
            })

        return pd.DataFrame(prompt_metrics)
    except Exception:
        return pd.DataFrame()


def render_prompt_lab_page(df: pd.DataFrame):
    """Page for understanding and comparing prompts (hillclimbing)."""

    render_page_header(
        "Prompt Lab",
        "Understand how prompt engineering affects evaluation quality"
    )

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <a href="https://github.com/rasiulyte/evaluation-system/blob/main/docs/PROMPTING_STRATEGIES.md" target="_blank"
           style="color: {COLORS['teal']}; text-decoration: none; font-size: 0.9rem;">
            📄 View full documentation on GitHub →
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Educational intro
    st.markdown(f"""
    <div class="metric-card" style="padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.1rem; color: {COLORS['navy']}; margin-bottom: 1rem; font-weight: 500;">
            What is Hillclimbing?
        </div>
        <div style="color: {COLORS['charcoal']}; line-height: 1.7;">
            <strong>Hillclimbing</strong> is an iterative optimization technique where you make small changes to a prompt
            and measure if the results improve. Like climbing a hill in fog — you take a step, check if you went up,
            and keep going in the direction that improves your metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # The process
    render_section_header("The Hillclimbing Process")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 1.5rem; color: {COLORS['teal']}; margin-bottom: 0.5rem;">1</div>
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Baseline</div>
            <div style="font-size: 0.85rem; color: {COLORS['charcoal']};">Run evaluation with current prompt</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 1.5rem; color: {COLORS['teal']}; margin-bottom: 0.5rem;">2</div>
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Modify</div>
            <div style="font-size: 0.85rem; color: {COLORS['charcoal']};">Make a targeted change to the prompt</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 1.5rem; color: {COLORS['teal']}; margin-bottom: 0.5rem;">3</div>
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Measure</div>
            <div style="font-size: 0.85rem; color: {COLORS['charcoal']};">Run evaluation with new prompt</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 1.5rem; color: {COLORS['teal']}; margin-bottom: 0.5rem;">4</div>
            <div style="font-weight: 500; color: {COLORS['navy']}; margin-bottom: 0.5rem;">Compare</div>
            <div style="font-size: 0.85rem; color: {COLORS['charcoal']};">Keep the change if metrics improve</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Prompt Version Analysis
    render_section_header("Prompt Version Analysis")

    st.markdown("Each prompt version represents a different approach. Click to see details, use cases, and limitations.")

    prompts = load_all_prompts()

    # v1 Zero Shot
    with st.expander("**v1_zero_shot** — Basic approach, no examples"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** Simply ask the model to classify without examples or detailed instructions.

            **✓ Use when:**
            - Quick baseline testing
            - Evaluating model's inherent capability
            - Token budget is very limited

            **✗ Don't use when:**
            - You need consistent output format
            - You need confidence scores
            - High accuracy is required

            **Limitations:**
            - Output format varies (hard to parse)
            - No confidence scores
            - Model may misunderstand the task
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Baselines</div>
            </div>
            """, unsafe_allow_html=True)
        if "v1_zero_shot" in prompts:
            st.code(prompts["v1_zero_shot"], language=None)

    # v2 Few Shot
    with st.expander("**v2_few_shot** — Learning from examples"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** Provide examples of correct classifications to guide the model.

            **✓ Use when:**
            - Model struggles with zero-shot
            - You have good representative examples
            - Consistency matters more than token cost

            **✗ Don't use when:**
            - Token budget is tight (examples add tokens)
            - Your examples might bias edge cases
            - You need structured output

            **Limitations:**
            - Examples may not cover all cases
            - Higher token usage
            - Still no guaranteed output format
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Consistency</div>
            </div>
            """, unsafe_allow_html=True)
        if "v2_few_shot" in prompts:
            st.code(prompts["v2_few_shot"], language=None)

    # v3 Chain of Thought
    with st.expander("**v3_chain_of_thought** — Step-by-step reasoning"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** Ask the model to reason through the problem step by step before concluding.

            **✓ Use when:**
            - Complex cases requiring nuanced judgment
            - You want to understand the reasoning
            - Debugging why classifications fail

            **✗ Don't use when:**
            - Speed/latency is critical
            - Token budget is limited
            - You only need the final answer

            **Limitations:**
            - Slower (more tokens generated)
            - Reasoning can still be flawed
            - Output format still varies
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Explainability</div>
            </div>
            """, unsafe_allow_html=True)
        if "v3_chain_of_thought" in prompts:
            st.code(prompts["v3_chain_of_thought"], language=None)

    # v4 Rubric Based
    with st.expander("**v4_rubric_based** — Multi-dimensional scoring"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** Score across 4 dimensions (grounding, factual, inference, numeric) with a 100-point scale.

            **✓ Use when:**
            - You need detailed breakdown of WHY something is hallucinated
            - Analyzing failure patterns
            - Training human reviewers

            **✗ Don't use when:**
            - You need automated metric calculation
            - Simple pass/fail is sufficient
            - Parsing reliability matters

            **Limitations:**
            - Complex text output (hard to parse)
            - No single confidence score for correlation metrics
            - Model may not follow rubric consistently
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Deep Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        if "v4_rubric_based" in prompts:
            st.code(prompts["v4_rubric_based"], language=None)

    # v5 Structured Output
    with st.expander("**v5_structured_output** — JSON format for automation"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** Request JSON output with classification, confidence, and reasoning fields.

            **✓ Use when:**
            - Building automated pipelines
            - You need parseable output
            - Integrating with other systems

            **✗ Don't use when:**
            - You need meaningful confidence scores (use v6)
            - Correlation metrics matter (Spearman ~0.26)

            **Limitations:**
            - Confidence values are arbitrary (not calibrated)
            - Model picks numbers without clear meaning
            - Spearman correlation ~0.26 (nearly random)
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card status-warning">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Automation</div>
                <div style="font-size: 0.75rem; color: {COLORS['amber']}; margin-top: 0.25rem;">⚠ Poor calibration</div>
            </div>
            """, unsafe_allow_html=True)
        if "v5_structured_output" in prompts:
            st.code(prompts["v5_structured_output"], language=None)

    # v6 Calibrated Confidence
    with st.expander("**v6_calibrated_confidence** — JSON with meaningful confidence ⭐ Recommended"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Approach:** JSON output with explicit calibration guidelines for confidence values.

            **✓ Use when:**
            - Production systems
            - You need trustworthy confidence scores
            - Correlation metrics matter
            - You want to threshold by confidence

            **✗ Don't use when:**
            - You need detailed dimensional breakdown (use v4)
            - Explainability is the primary goal (use v3)

            **Why it's recommended:**
            - Confidence actually predicts correctness
            - Spearman correlation ~0.84 (vs 0.26 for v5)
            - Easy to parse AND meaningful scores
            """)
        with col2:
            st.markdown(f"""
            <div class="metric-card status-good">
                <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">Best for</div>
                <div style="font-weight: 500; color: {COLORS['navy']};">Production</div>
                <div style="font-size: 0.75rem; color: {COLORS['good']}; margin-top: 0.25rem;">⭐ Recommended</div>
            </div>
            """, unsafe_allow_html=True)
        if "v6_calibrated_confidence" in prompts:
            st.code(prompts["v6_calibrated_confidence"], language=None)

    st.markdown("---")

    # Comparison table
    render_section_header("Quick Comparison")

    st.markdown(f"""
    <div class="metric-card">
        <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid {COLORS['light_gray']};">
                <th style="text-align: left; padding: 0.5rem;">Version</th>
                <th style="text-align: left; padding: 0.5rem;">Output</th>
                <th style="text-align: left; padding: 0.5rem;">Parseable</th>
                <th style="text-align: left; padding: 0.5rem;">Confidence</th>
                <th style="text-align: left; padding: 0.5rem;">Best For</th>
            </tr>
            <tr style="border-bottom: 1px solid {COLORS['light_gray']};">
                <td style="padding: 0.5rem;"><code>v1</code></td>
                <td style="padding: 0.5rem;">Text</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">Baselines</td>
            </tr>
            <tr style="border-bottom: 1px solid {COLORS['light_gray']};">
                <td style="padding: 0.5rem;"><code>v2</code></td>
                <td style="padding: 0.5rem;">Text</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">Consistency</td>
            </tr>
            <tr style="border-bottom: 1px solid {COLORS['light_gray']};">
                <td style="padding: 0.5rem;"><code>v3</code></td>
                <td style="padding: 0.5rem;">Text + reasoning</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">❌</td>
                <td style="padding: 0.5rem;">Explainability</td>
            </tr>
            <tr style="border-bottom: 1px solid {COLORS['light_gray']};">
                <td style="padding: 0.5rem;"><code>v4</code></td>
                <td style="padding: 0.5rem;">Rubric scores</td>
                <td style="padding: 0.5rem;">⚠️</td>
                <td style="padding: 0.5rem;">⚠️ (derived)</td>
                <td style="padding: 0.5rem;">Deep analysis</td>
            </tr>
            <tr style="border-bottom: 1px solid {COLORS['light_gray']};">
                <td style="padding: 0.5rem;"><code>v5</code></td>
                <td style="padding: 0.5rem;">JSON</td>
                <td style="padding: 0.5rem;">✅</td>
                <td style="padding: 0.5rem;">⚠️ (uncalibrated)</td>
                <td style="padding: 0.5rem;">Automation</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;"><code>v6</code> ⭐</td>
                <td style="padding: 0.5rem;">JSON</td>
                <td style="padding: 0.5rem;">✅</td>
                <td style="padding: 0.5rem;">✅ (calibrated)</td>
                <td style="padding: 0.5rem;">Production</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # How to interpret
    render_section_header("Interpreting Hillclimbing Results")

    st.markdown(f"""
    <div class="metric-card">
        <div style="color: {COLORS['charcoal']}; line-height: 1.8;">
            <strong>When comparing prompts, look for:</strong>
            <table style="width: 100%; margin-top: 1rem; font-size: 0.9rem;">
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Metric</strong></td>
                    <td style="padding: 0.5rem 0;"><strong>What Improvement Means</strong></td>
                </tr>
                <tr style="border-top: 1px solid {COLORS['light_gray']};">
                    <td style="padding: 0.5rem 0;">F1 Score ↑</td>
                    <td style="padding: 0.5rem 0;">Better balance of catching hallucinations without false alarms</td>
                </tr>
                <tr style="border-top: 1px solid {COLORS['light_gray']};">
                    <td style="padding: 0.5rem 0;">Spearman ↑</td>
                    <td style="padding: 0.5rem 0;">Confidence scores are more meaningful/trustworthy</td>
                </tr>
                <tr style="border-top: 1px solid {COLORS['light_gray']};">
                    <td style="padding: 0.5rem 0;">Bias → 0</td>
                    <td style="padding: 0.5rem 0;">System is more balanced (not too aggressive or lenient)</td>
                </tr>
                <tr style="border-top: 1px solid {COLORS['light_gray']};">
                    <td style="padding: 0.5rem 0;">MAE ↓</td>
                    <td style="padding: 0.5rem 0;">Confidence values are better calibrated to actual correctness</td>
                </tr>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Tips
    render_section_header("Hillclimbing Tips")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: 500; color: {COLORS['good']}; margin-bottom: 0.5rem;">✓ Do</div>
            <ul style="color: {COLORS['charcoal']}; font-size: 0.9rem; margin: 0; padding-left: 1.25rem;">
                <li>Change one thing at a time</li>
                <li>Run on the same test cases</li>
                <li>Use enough samples (20+)</li>
                <li>Track which changes helped</li>
                <li>Consider multiple metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-weight: 500; color: {COLORS['poor']}; margin-bottom: 0.5rem;">✗ Don't</div>
            <ul style="color: {COLORS['charcoal']}; font-size: 0.9rem; margin: 0; padding-left: 1.25rem;">
                <li>Change multiple things at once</li>
                <li>Use different test cases</li>
                <li>Optimize for just one metric</li>
                <li>Ignore confidence calibration</li>
                <li>Stop after first improvement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


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

    # Initialize session state for navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Getting Started"

    # Sidebar
    with st.sidebar:
        # Logo - clickable, styled via CSS
        if st.button("◈ Eval Lab", key="logo_btn", use_container_width=True):
            st.session_state.current_page = "Getting Started"
            st.rerun()

        st.markdown(f"""
        <div style="padding: 0 0 0.75rem 0; margin-top: -0.5rem;">
            <div style="font-size: 0.8rem; color: {COLORS['medium_gray']};">
                Learning Sandbox
            </div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem;">
                <a href="https://rasar.ai" target="_blank" style="color: {COLORS['teal']};">rasar.ai</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Grouped Navigation with colorful icons
        nav_groups = {
            "LEARN": {
                "icon": "◇",
                "color": "#8b5cf6",  # Purple
                "pages": ["Getting Started", "Failure Modes", "Prompt Lab", "Understanding Metrics", "Limitations"]
            },
            "ANALYZE": {
                "icon": "◈",
                "color": "#f59e0b",  # Amber
                "pages": ["Metrics Overview", "Slice Analysis", "Trends", "Compare Runs", "Run History", "Test Cases"]
            },
            "RUN": {
                "icon": "▷",
                "color": "#10b981",  # Green
                "pages": ["Run Evaluation"]
            }
        }

        # Build flat list for routing
        all_pages = []
        for group_data in nav_groups.values():
            all_pages.extend(group_data["pages"])

        # Render grouped navigation with colored icons and styled headers
        for group_name, group_data in nav_groups.items():
            icon = group_data["icon"]
            color = group_data["color"]
            css_class = group_name.lower()  # learn, analyze, run
            st.markdown(f'''
            <div class="nav-section-header {css_class}">
                <span style="color: {color}; font-size: 1rem; margin-right: 6px;">{icon}</span>{group_name}
            </div>
            ''', unsafe_allow_html=True)

            for page_name in group_data["pages"]:
                is_active = st.session_state.current_page == page_name
                indicator = "●" if is_active else "○"

                # Use primary type for active to enable CSS styling
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{indicator} {page_name}",
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.current_page = page_name
                    st.rerun()

        page = st.session_state.current_page

        # Bottom metadata area
        st.markdown('<div class="nav-metadata">', unsafe_allow_html=True)

        if not df.empty:
            total_runs = df["run_id"].nunique()
            latest_run = df.sort_values("timestamp", ascending=False).iloc[0]
            latest_date, _ = to_pst(latest_run['timestamp'])
            st.markdown(f"""
            <div style="font-size: 13px; color: #6b7280; line-height: 1.6;">
                <div><strong>{total_runs}</strong> evaluation runs</div>
                <div style="margin-top: 4px;">Latest: {latest_date}</div>
            </div>
            """, unsafe_allow_html=True)

        # Database info (collapsed)
        with st.expander("System", expanded=False):
            try:
                debug = db.debug_info()
                st.caption(f"Backend: {debug.get('backend', 'Unknown')}")
                st.caption(f"Metrics: {debug.get('metrics_count', 0)}")
                st.caption(f"Test Results: {debug.get('test_results_count', 0)}")
            except Exception as e:
                st.caption(f"Error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Route to pages
    if page == "Getting Started":
        render_home_page()
    elif page == "Metrics Overview":
        render_metrics_overview_page(df)
    elif page == "Slice Analysis":
        render_slice_analysis_page(df)
    elif page == "Trends":
        render_trends_page(df)
    elif page == "Compare Runs":
        render_compare_runs_page(df)
    elif page == "Run History":
        render_run_history_page(df)
    elif page == "Test Cases":
        render_test_cases_page()
    elif page == "Failure Modes":
        render_failure_modes_page()
    elif page == "Limitations":
        render_limitations_page()
    elif page == "Prompt Lab":
        render_prompt_lab_page(df)
    elif page == "Run Evaluation":
        render_run_evaluation_page()
    elif page == "Understanding Metrics":
        render_guide_page()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
