"""Streamlit app for a household energy consumption regression project."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "household_energy_consumption.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.csv"
LEGACY_METRICS_PATH = PROJECT_ROOT / "results" / "test_models_metrics.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"
TARGET_COLUMN = "Energy_Consumption_kWh"
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "Household_ID",
    "Date",
    TARGET_COLUMN,
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC",
]

FEATURE_COLUMNS = [
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC_Binary",
    "temperature_x_ac",
    "household_size_x_ac",
]

MODEL_OPTIONS = [
    "Random Forest optimisé Optuna",
    "Random Forest Regressor",
    "Linear Regression",
]
COLORWAY = ["#14B8A6", "#F97316", "#3B82F6", "#A855F7", "#E11D48", "#84CC16"]
MODEL_NAME_LABELS = {
    "baseline_linear_regression": "Régression linéaire",
    "random_forest": "Random Forest",
    "random_forest_optuna": "Random Forest optimisé Optuna",
}
DEFAULT_MODEL_METRICS = pd.DataFrame(
    [
        {
            "model": "random_forest_optuna",
            "mae": 1.1098651063244223,
            "mse": 2.0512860853837047,
            "rmse": 1.4322311564072696,
            "r2": 0.9327487963978991,
        },
        {
            "model": "random_forest",
            "mae": 1.1187209078278832,
            "mse": 2.0835413547364894,
            "rmse": 1.443447731903199,
            "r2": 0.9316913107054151,
        },
        {
            "model": "baseline_linear_regression",
            "mae": 1.1784634224446437,
            "mse": 2.1790663028722514,
            "rmse": 1.4761660824149332,
            "r2": 0.9285595350930648,
        },
    ]
)
NAVIGATION_PAGES = [
    "Accueil",
    "Données & EDA",
    "Modélisation",
    "Prédiction",
]


def inject_css() -> None:
    """Inject a premium visual system for the Streamlit interface."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #101828;
            --muted: #667085;
            --surface: #ffffff;
            --line: #e4e7ec;
            --teal: #0f766e;
            --orange: #ea580c;
            --blue: #2563eb;
            --purple: #7c3aed;
            --rose: #be123c;
            --green: #65a30d;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f6fbff 0%, #ffffff 34%, #f8fafc 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            height: 0;
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        h2 {
            margin-top: 1.2rem;
        }

        .top-bar {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.96)),
                linear-gradient(90deg, rgba(20, 184, 166, 0.13), rgba(249, 115, 22, 0.13), rgba(124, 58, 237, 0.13));
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-top: 6px solid #14B8A6;
            border-radius: 8px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.11);
            padding: 1.15rem 1.2rem 0.95rem 1.2rem;
            margin-bottom: 0.95rem;
            position: relative;
            overflow: hidden;
        }

        .top-bar::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 6px;
            background: linear-gradient(90deg, #14B8A6, #F97316, #3B82F6, #A855F7, #E11D48);
        }

        .top-bar-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: white;
            font-weight: 900;
            background: linear-gradient(135deg, #14B8A6, #2563EB 52%, #A855F7);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
        }

        .brand-title {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 850;
            line-height: 1.15;
            margin: 0;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.86rem;
            margin: 0.15rem 0 0 0;
        }

        .target-pill {
            color: #0f172a;
            background: #ecfeff;
            border: 1px solid #bae6fd;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-size: 0.84rem;
            font-weight: 750;
            white-space: nowrap;
        }

        div[data-testid="stRadio"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.35rem;
            margin-top: 0.75rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
        }

        div[data-testid="stRadio"] > label {
            display: none;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.35rem;
        }

        div[data-testid="stRadio"] label {
            background: #f8fafc;
            border: 1px solid transparent;
            border-radius: 8px;
            min-height: 42px;
            padding: 0.55rem 0.7rem;
            justify-content: center;
            transition: all 160ms ease;
        }

        div[data-testid="stRadio"] label:hover {
            border-color: #cbd5e1;
            background: #f1f5f9;
        }

        div[data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(135deg, #0f766e, #2563eb);
            border-color: transparent;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
        }

        div[data-testid="stRadio"] label:has(input:checked) p {
            color: white;
            font-weight: 800;
        }

        div[data-testid="stRadio"] label p {
            color: #334155;
            font-weight: 700;
            text-align: center;
            line-height: 1.15;
            white-space: normal;
        }

        div[data-testid="stRadio"] label span:first-child {
            display: none;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
        }

        @media (max-width: 800px) {
            div[data-testid="stRadio"] div[role="radiogroup"] {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .target-pill {
                width: 100%;
                text-align: center;
            }
        }

        .hero {
            background:
                linear-gradient(120deg, rgba(15, 23, 42, 0.96), rgba(15, 118, 110, 0.92) 46%, rgba(124, 58, 237, 0.86)),
                radial-gradient(circle at 78% 12%, rgba(249, 115, 22, 0.45), transparent 30%);
            color: white;
            padding: 2.35rem 2.45rem;
            border-radius: 8px;
            box-shadow: 0 22px 48px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            font-size: clamp(2.15rem, 4.8vw, 4.2rem);
            line-height: 1.02;
            margin: 0;
            max-width: 980px;
        }

        .hero p {
            color: rgba(255, 255, 255, 0.86);
            font-size: 1.08rem;
            max-width: 880px;
            margin: 0.85rem 0 0 0;
        }

        .eyebrow {
            display: inline-flex;
            gap: 0.4rem;
            align-items: center;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.24);
            color: white;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 750;
            margin-bottom: 0.85rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            min-height: 116px;
        }

        .metric-card .label {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.36rem;
        }

        .metric-card .value {
            color: var(--ink);
            font-size: 1.6rem;
            font-weight: 780;
            line-height: 1.15;
        }

        .metric-card .hint {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.36rem;
        }

        .insight {
            background: white;
            border: 1px solid var(--line);
            border-left: 5px solid var(--teal);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
            min-height: 128px;
        }

        .insight.orange { border-left-color: var(--orange); }
        .insight.blue { border-left-color: var(--blue); }
        .insight.purple { border-left-color: var(--purple); }
        .insight.rose { border-left-color: var(--rose); }

        .insight strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .insight span {
            color: var(--muted);
            font-size: 0.94rem;
        }

        .visual-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            min-height: 170px;
        }

        .visual-card .card-icon {
            width: 38px;
            height: 38px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-weight: 850;
            font-size: 0.78rem;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #14B8A6, #2563EB);
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
        }

        .visual-card.orange .card-icon {
            background: linear-gradient(135deg, #F97316, #E11D48);
        }

        .visual-card.purple .card-icon {
            background: linear-gradient(135deg, #3B82F6, #A855F7);
        }

        .visual-card strong {
            display: block;
            color: var(--ink);
            font-size: 1.04rem;
            margin-bottom: 0.4rem;
        }

        .visual-card span {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .question-card {
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.93) 52%, rgba(124, 58, 237, 0.9));
            color: #ffffff;
            border-radius: 8px;
            padding: 1.45rem 1.55rem;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.2);
            margin: 1.1rem 0 1.25rem 0;
        }

        .question-card .label {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .question-card .question {
            font-size: clamp(1.25rem, 2.4vw, 1.9rem);
            line-height: 1.22;
            font-weight: 850;
            max-width: 1000px;
        }

        .info-banner {
            background: linear-gradient(135deg, #0f766e, #2563eb);
            color: #ffffff;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18);
            margin: 0.85rem 0 1rem 0;
            font-weight: 760;
        }

        .data-preview-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            margin: 1.15rem 0;
        }

        .data-preview-card .preview-title {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }

        .data-preview-card .preview-text {
            color: var(--muted);
            margin-bottom: 0.8rem;
            font-size: 0.95rem;
        }

        .plot-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
            margin: 1rem 0 1.35rem 0;
        }

        .plot-card-title {
            color: var(--ink);
            font-size: 1.28rem;
            font-weight: 850;
            margin-bottom: 0.35rem;
        }

        .plot-card-text {
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.5;
            margin-bottom: 1rem;
            max-width: 980px;
        }

        .plot-card img {
            display: block;
            width: 100%;
            max-width: 1120px;
            height: auto;
            margin: 0 auto;
            border-radius: 8px;
            border: 1px solid #eef2f7;
        }

        .model-conclusion {
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.94) 52%, rgba(124, 58, 237, 0.92));
            color: #ffffff;
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 40px rgba(37, 99, 235, 0.2);
            margin: 1rem 0 1.2rem 0;
        }

        .model-conclusion .label {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            padding: 0.24rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .model-conclusion strong {
            display: block;
            font-size: 1.45rem;
            margin-bottom: 0.4rem;
        }

        .model-conclusion span {
            display: block;
            color: rgba(255, 255, 255, 0.88);
            line-height: 1.55;
            max-width: 1060px;
        }

        .prediction-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem 1.25rem 0.95rem 1.25rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
            margin: 1.15rem 0 1.25rem 0;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
        }

        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] p,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] p {
            color: #1f2937 !important;
            font-weight: 760 !important;
            opacity: 1 !important;
        }

        div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
            color: #1f2937 !important;
            opacity: 1 !important;
        }

        div[data-testid="stSlider"] div {
            color: #1f2937;
        }

        div[data-testid="stAlert"] {
            background: #dbeafe;
            border: 1px solid #93c5fd;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            color: #0f172a;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
        }

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div {
            color: #0f172a !important;
            opacity: 1 !important;
            font-weight: 650;
        }

        .section-note {
            color: var(--muted);
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }

        .footer {
            color: #667085;
            border-top: 1px solid var(--line);
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
            font-size: 0.9rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 750;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply one consistent Plotly style to all charts."""

    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        height=height,
        margin=dict(l=10, r=10, t=58, b=20),
        title_font=dict(size=18, color="#101828"),
        font=dict(color="#1f2937", size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e4e7ec",
            borderwidth=1,
            font=dict(color="#1f2937"),
        ),
        coloraxis_colorbar=dict(
            title_font=dict(color="#1f2937", size=14),
            tickfont=dict(color="#1f2937", size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color="#1f2937", size=13),
        title_font=dict(color="#1f2937", size=14),
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color="#1f2937", size=13),
        title_font=dict(color="#1f2937", size=14),
    )
    return fig


def render_metric_card(label: str, value: str, hint: str) -> None:
    """Render a compact KPI card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(title: str, body: str, color: str = "") -> None:
    """Render a short professional content block."""

    st.markdown(
        f"""
        <div class="insight {color}">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visual_card(icon: str, title: str, body: str, color: str = "") -> None:
    """Render a visual home-page card with a compact icon marker."""

    st.markdown(
        f"""
        <div class="visual-card {color}">
            <div class="card-icon">{icon}</div>
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot_card(title: str, interpretation: str, image_path: Path) -> None:
    """Render an EDA plot image inside a styled card."""

    if not image_path.exists():
        st.warning(f"Graphique introuvable : {image_path.name}")
        return

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="plot-card">
            <div class="plot-card-title">{html.escape(title)}</div>
            <div class="plot-card-text">{html.escape(interpretation)}</div>
            <img src="data:image/png;base64,{image_base64}" alt="{html.escape(title)}">
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_model_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return model metrics formatted for display in the app."""

    expected_columns = ["model", "mae", "mse", "rmse", "r2"]
    available_columns = [column for column in expected_columns if column in metrics_df.columns]
    display_df = metrics_df[available_columns].copy()

    if "model" in display_df:
        display_df["Modèle testé"] = display_df["model"].map(MODEL_NAME_LABELS).fillna(
            display_df["model"]
        )
        display_df = display_df.drop(columns=["model"])

    rename_map = {
        "mae": "MAE",
        "mse": "MSE",
        "rmse": "RMSE",
        "r2": "R2",
    }
    display_df = display_df.rename(columns=rename_map)
    ordered_columns = [
        column
        for column in ["Modèle testé", "MAE", "MSE", "RMSE", "R2"]
        if column in display_df.columns
    ]
    display_df = display_df[ordered_columns]

    for column in ["MAE", "MSE", "RMSE", "R2"]:
        if column in display_df:
            display_df[column] = display_df[column].astype(float).map(lambda value: f"{value:.4f}")

    return display_df


def get_best_model_row(metrics_df: pd.DataFrame) -> pd.Series:
    """Return the best tested model using RMSE as the primary criterion."""

    if "rmse" not in metrics_df.columns:
        return DEFAULT_MODEL_METRICS.sort_values("rmse").iloc[0]
    return metrics_df.sort_values("rmse").iloc[0]


def render_model_conclusion(best_model: pd.Series) -> None:
    """Render the retained-model conclusion block."""

    model_key = str(best_model.get("model", "random_forest_optuna"))
    model_label = MODEL_NAME_LABELS.get(model_key, model_key)
    rmse = float(best_model.get("rmse", 1.4322311564072696))
    mae = float(best_model.get("mae", 1.1098651063244223))
    r2 = float(best_model.get("r2", 0.9327487963978991))

    st.markdown(
        f"""
        <div class="model-conclusion">
            <div class="label">Conclusion du choix modèle</div>
            <strong>Modèle retenu : {html.escape(model_label)}</strong>
            <span>
                Ce modèle est conservé car il obtient la meilleure erreur de prédiction
                sur le jeu de test : RMSE = {rmse:.3f} kWh, MAE = {mae:.3f} kWh
                et R2 = {r2:.3f}. Il combine donc une erreur moyenne faible avec une
                très forte capacité à expliquer la consommation énergétique observée.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float, digits: int = 0) -> str:
    """Format numbers for the French dashboard."""

    return f"{value:,.{digits}f}".replace(",", " ")


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Load the dataset bundled with the project."""

    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load saved model metrics when available."""

    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    if LEGACY_METRICS_PATH.exists():
        return pd.read_csv(LEGACY_METRICS_PATH)
    return DEFAULT_MODEL_METRICS.copy()


def load_dataset(uploaded_file: Any | None) -> pd.DataFrame:
    """Load the uploaded CSV file or fallback to the repository dataset."""

    if uploaded_file is None:
        return load_default_dataset()
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return required columns that are missing from the dataframe."""

    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and create model features."""

    missing_columns = validate_dataset(df)
    if missing_columns:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing_columns))

    prepared = df.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared["Has_AC"] = prepared["Has_AC"].astype(str).str.strip().str.title()
    prepared = prepared.dropna(
        subset=[
            TARGET_COLUMN,
            "Household_Size",
            "Avg_Temperature_C",
            "Has_AC",
        ]
    ).copy()
    prepared["Has_AC_Binary"] = prepared["Has_AC"].map({"Yes": 1, "No": 0})
    prepared = prepared.dropna(subset=["Has_AC_Binary"]).copy()
    prepared["Has_AC_Binary"] = prepared["Has_AC_Binary"].astype(int)
    prepared["temperature_x_ac"] = (
        prepared["Avg_Temperature_C"] * prepared["Has_AC_Binary"]
    )
    prepared["household_size_x_ac"] = (
        prepared["Household_Size"] * prepared["Has_AC_Binary"]
    )
    prepared["consumption_per_person"] = (
        prepared[TARGET_COLUMN] / prepared["Household_Size"].replace(0, np.nan)
    )
    return prepared


def split_dataset(
    prepared_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split model features and target into training and test sets."""

    X = prepared_df[FEATURE_COLUMNS]
    y = prepared_df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def build_model(model_name: str) -> Pipeline:
    """Create the selected regression pipeline."""

    if model_name == "Linear Regression":
        estimator = LinearRegression()
    elif model_name == "Random Forest Regressor":
        estimator = RandomForestRegressor(
            n_estimators=140,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif model_name == "Random Forest optimisé Optuna":
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=7,
            min_samples_leaf=6,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Modele non supporte : {model_name}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_cached_model(prepared_df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Train a model and return metrics, predictions and artifacts."""

    X_train, X_test, y_train, y_test = split_dataset(prepared_df)
    model = build_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "Reel": y_test.reset_index(drop=True),
            "Prediction": y_pred,
            "Erreur": y_test.reset_index(drop=True) - y_pred,
        }
    )
    return {
        "model_name": model_name,
        "model": model,
        "metrics": {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "R2": float(r2_score(y_test, y_pred)),
        },
        "predictions": predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def render_home(df: pd.DataFrame, prepared_df: pd.DataFrame) -> None:
    """Render the app landing page."""

    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">Machine Learning supervisé · Régression · Énergie résidentielle</span>
            <h1>Prédiction de la consommation énergétique des foyers</h1>
            <p>
                Aider un fournisseur d'énergie comme EDF à anticiper la demande
                des foyers, comprendre les pics de consommation et mieux accompagner
                les clients dans la maîtrise de leur facture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card(
            "Observations",
            format_number(float(len(df))),
            "Base d'analyse pour comprendre les comportements de consommation.",
        )
    with col_b:
        render_metric_card(
            "Foyers",
            format_number(float(df["Household_ID"].nunique())),
            "Clients avec des profils de consommation différents.",
        )
    with col_c:
        render_metric_card(
            "Conso. moyenne",
            f"{prepared_df[TARGET_COLUMN].mean():.2f} kWh",
            "Consommation moyenne observée par foyer.",
        )
    with col_d:
        render_metric_card(
            "Avec climatisation",
            f"{prepared_df['Has_AC_Binary'].mean() * 100:.1f} %",
            "Facteur pouvant influencer la consommation lors des périodes chaudes.",
        )

    st.subheader("Le problème business")
    col_context, col_problem, col_goal = st.columns(3)
    with col_context:
        render_visual_card(
            "01",
            "Contexte actuel",
            "Les prix de l'énergie augmentent, les usages évoluent avec le "
            "télétravail, les appareils connectés et la climatisation, ce qui "
            "rend la consommation des foyers plus difficile à prévoir.",
            "blue",
        )
    with col_problem:
        render_visual_card(
            "02",
            "Problème pour EDF",
            "Tous les foyers ne consomment pas de la même manière. Une famille "
            "nombreuse, un foyer équipé d'une climatisation ou un client consommant "
            "fortement en heures de pointe peuvent créer des profils de demande "
            "très différents.",
            "orange",
        )
    with col_goal:
        render_visual_card(
            "03",
            "Enjeu du projet",
            "L'objectif est d'estimer la consommation énergétique en kWh afin de "
            "mieux comprendre les facteurs qui influencent la demande et d'aider "
            "à anticiper les pics de consommation.",
            "purple",
        )

    st.markdown(
        """
        <div class="question-card">
            <div class="label">Question centrale</div>
            <div class="question">
                Peut-on prédire la consommation énergétique d'un foyer à partir
                de ses caractéristiques et de ses habitudes de consommation ?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Ce que permet l'application")
    col_explore, col_compare, col_predict = st.columns(3)
    with col_explore:
        render_visual_card(
            "EDA",
            "Explorer les données",
            "Comprendre la structure du dataset et les profils de consommation.",
        )
    with col_compare:
        render_visual_card(
            "ML",
            "Comparer les modèles",
            "Évaluer les performances d'une régression linéaire et d'un Random Forest.",
            "purple",
        )
    with col_predict:
        render_visual_card(
            "kWh",
            "Faire une prédiction",
            "Estimer la consommation énergétique d'un foyer à partir de ses caractéristiques.",
            "orange",
        )


def render_data_page(df: pd.DataFrame, prepared_df: pd.DataFrame, uploaded_file: Any | None) -> None:
    """Render data exploration and quality checks."""

    st.title("Exploration des données")
    st.markdown(
        "<p class='section-note'>Comprendre la structure du dataset, la qualité des données et les premiers comportements de consommation.</p>",
        unsafe_allow_html=True,
    )

    dataset_name = DATA_PATH.name if uploaded_file is None else uploaded_file.name
    st.markdown(
        f"""
        <div class="info-banner">
            Dataset utilisé : {dataset_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card(
            "Lignes",
            format_number(float(len(df))),
            "Nombre total d'observations disponibles.",
        )
    with col_b:
        render_metric_card(
            "Colonnes",
            format_number(float(df.shape[1])),
            "Variables présentes dans le dataset.",
        )
    with col_c:
        render_metric_card(
            "Valeurs manquantes",
            format_number(float(df.isna().sum().sum())),
            "Contrôle du niveau de complétude.",
        )
    with col_d:
        render_metric_card(
            "Doublons",
            format_number(float(df.duplicated().sum())),
            "Détection des lignes répétées.",
        )

    st.markdown(
        """
        <div class="data-preview-card">
            <div class="preview-title">Aperçu du dataset</div>
            <div class="preview-text">
                Les premières lignes permettent de vérifier la structure des variables utilisées dans le modèle.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.subheader("Visualisations principales")
    st.markdown(
        """
        <p class='section-note'>
            Les graphiques suivants résument les premières tendances observées dans les données :
            distribution de la consommation, effet de la taille du foyer, rôle de la climatisation
            et profils moyens des foyers.
        </p>
        """,
        unsafe_allow_html=True,
    )

    render_plot_card(
        "Distribution de la consommation",
        "Ce graphique permet d'observer la répartition globale de la consommation énergétique, "
        "la consommation pendant les heures de pointe et la consommation moyenne par personne.",
        PLOTS_DIR / "distributions_consommation.png",
    )
    render_plot_card(
        "Effet de la taille du foyer et de la climatisation",
        "Ces visualisations montrent que la consommation augmente généralement avec la taille "
        "du foyer et que la présence de climatisation peut modifier le niveau moyen de consommation.",
        PLOTS_DIR / "consommation_taille_climatisation.png",
    )
    render_plot_card(
        "Profils moyens des foyers",
        "Cette analyse permet d'observer la consommation moyenne par foyer ainsi que la volatilité "
        "des comportements de consommation.",
        PLOTS_DIR / "profils_foyers.png",
    )

    quality = pd.DataFrame(
        {
            "colonne": df.columns,
            "type": df.dtypes.astype(str).values,
            "valeurs_manquantes": df.isna().sum().values,
            "valeurs_uniques": df.nunique().values,
        }
    )
    with st.expander("Contrôle qualité détaillé", expanded=False):
        st.dataframe(quality, use_container_width=True, hide_index=True)


def render_model_page(prepared_df: pd.DataFrame) -> None:
    """Render training, metrics and model performance."""

    st.header("Modélisation et performance")
    st.markdown(
        "<p class='section-note'>Comparer les trois modèles testés, expliquer les métriques utilisées et présenter le modèle retenu.</p>",
        unsafe_allow_html=True,
    )

    saved_metrics = load_metrics()
    display_metrics = prepare_model_metrics_table(saved_metrics)

    st.subheader("Modèles testés")
    st.markdown(
        """
        <p class='section-note'>
            Trois approches ont été comparées sur le même jeu de test afin de choisir
            le modèle le plus fiable pour prédire <strong>Energy_Consumption_kWh</strong>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    st.subheader("Métriques utilisées")
    col_mae, col_mse, col_rmse, col_r2 = st.columns(4)
    with col_mae:
        render_visual_card(
            "MAE",
            "Erreur moyenne",
            "Mesure l'écart moyen en kWh. Elle est simple à expliquer en présentation.",
        )
    with col_mse:
        render_visual_card(
            "MSE",
            "Pénaliser les fortes erreurs",
            "Amplifie les erreurs importantes pour repérer les modèles moins stables.",
            "blue",
        )
    with col_rmse:
        render_visual_card(
            "RMSE",
            "Critère principal",
            "Reste exprimée en kWh et pénalise les grosses erreurs. Plus elle est basse, mieux c'est.",
            "purple",
        )
    with col_r2:
        render_visual_card(
            "R2",
            "Variance expliquée",
            "Indique la part de la consommation expliquée par le modèle. Plus il est proche de 1, mieux c'est.",
            "orange",
        )

    best_model = get_best_model_row(saved_metrics)
    render_model_conclusion(best_model)

    st.subheader("Démonstration interactive du modèle")
    st.markdown(
        "<p class='section-note'>Cette zone permet de réentraîner rapidement un modèle et de visualiser ses erreurs sur le jeu de test.</p>",
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox("Modèle à entraîner", MODEL_OPTIONS)
    if st.button("Entraîner le modèle", type="primary"):
        with st.spinner(f"Entraînement du modèle {selected_model}..."):
            st.session_state["training_result"] = train_cached_model(prepared_df, selected_model)

    if "training_result" not in st.session_state:
        with st.spinner("Préparation du modèle de référence..."):
            st.session_state["training_result"] = train_cached_model(
                prepared_df, "Random Forest optimisé Optuna"
            )

    training_result = st.session_state["training_result"]
    metrics = training_result["metrics"]

    st.success(f"Modèle actif : {training_result['model_name']}")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card("MAE", f"{metrics['MAE']:.3f}", "Erreur absolue moyenne")
    with col_b:
        render_metric_card("RMSE", f"{metrics['RMSE']:.3f}", "Erreur type en kWh")
    with col_c:
        render_metric_card("R2", f"{metrics['R2']:.3f}", "Variance expliquée")
    with col_d:
        render_metric_card(
            "Test set",
            format_number(float(len(training_result["predictions"]))),
            "Observations évaluées",
        )

    prediction_df = training_result["predictions"]
    sample_df = prediction_df.sample(min(4500, len(prediction_df)), random_state=RANDOM_STATE)
    col_left, col_right = st.columns(2)
    with col_left:
        fig = px.scatter(
            sample_df,
            x="Reel",
            y="Prediction",
            opacity=0.58,
            color_discrete_sequence=["#14B8A6"],
            title="Prédictions vs valeurs réelles",
            labels={"Reel": "Valeur réelle (kWh)", "Prediction": "Prédiction (kWh)"},
        )
        min_value = float(min(prediction_df["Reel"].min(), prediction_df["Prediction"].min()))
        max_value = float(max(prediction_df["Reel"].max(), prediction_df["Prediction"].max()))
        fig.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(color="#F97316", width=2),
        )
        fig = apply_plot_style(fig, 400)
        fig.update_traces(marker=dict(color="#14B8A6", opacity=0.72))
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_right:
        fig = px.histogram(
            prediction_df,
            x="Erreur",
            nbins=50,
            color_discrete_sequence=["#7C3AED"],
            title="Distribution des erreurs résiduelles",
            labels={"Erreur": "Erreur réelle - prédiction (kWh)"},
        )
        fig = apply_plot_style(fig, 400)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    estimator = training_result["model"].named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Variable": FEATURE_COLUMNS,
                "Importance": estimator.feature_importances_,
            }
        ).sort_values("Importance", ascending=True)
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Variable",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Importance des variables du modèle",
        )
        fig = apply_plot_style(fig, 430)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


def make_prediction_row(
    household_size: int,
    avg_temperature: float,
    has_ac: bool,
) -> pd.DataFrame:
    """Create a one-row dataframe matching the training features."""

    has_ac_binary = int(has_ac)
    return pd.DataFrame(
        [
            {
                "Household_Size": household_size,
                "Avg_Temperature_C": avg_temperature,
                "Has_AC_Binary": has_ac_binary,
                "temperature_x_ac": avg_temperature * has_ac_binary,
                "household_size_x_ac": household_size * has_ac_binary,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def render_prediction_gauge(prediction: float, prepared_df: pd.DataFrame) -> None:
    """Render a gauge for the predicted energy consumption."""

    min_value = float(prepared_df[TARGET_COLUMN].min())
    max_value = float(prepared_df[TARGET_COLUMN].max())
    mean_value = float(prepared_df[TARGET_COLUMN].mean())
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " kWh"},
            gauge={
                "axis": {"range": [min_value, max_value]},
                "bar": {"color": "#14B8A6"},
                "steps": [
                    {"range": [min_value, mean_value], "color": "#dcfce7"},
                    {"range": [mean_value, max_value], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#F97316", "width": 3},
                    "thickness": 0.72,
                    "value": mean_value,
                },
            },
        )
    )
    st.plotly_chart(apply_plot_style(fig, 310), use_container_width=True)


def render_prediction_page(prepared_df: pd.DataFrame) -> None:
    """Render the model prediction form."""

    st.header("Prédiction de consommation")
    st.markdown(
        "<p class='section-note'>Simulez le profil d'un foyer et estimez sa consommation énergétique.</p>",
        unsafe_allow_html=True,
    )

    if "training_result" not in st.session_state:
        with st.spinner("Entraînement automatique du modèle de référence..."):
            st.session_state["training_result"] = train_cached_model(
                prepared_df, "Random Forest optimisé Optuna"
            )

    st.markdown(
        """
        <div class="prediction-panel">
            <div class="plot-card-title">Paramètres du foyer</div>
            <div class="plot-card-text">
                Ajustez les caractéristiques du foyer pour estimer sa consommation énergétique.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            household_size = st.slider("Taille du foyer", 1, 8, 4)
            has_ac = st.selectbox("Climatisation", ["Oui", "Non"]) == "Oui"
        with col_b:
            avg_temperature = st.slider(
                "Température moyenne (C)",
                float(np.floor(prepared_df["Avg_Temperature_C"].min())),
                float(np.ceil(prepared_df["Avg_Temperature_C"].max())),
                float(round(prepared_df["Avg_Temperature_C"].median(), 1)),
                step=0.1,
            )
        submitted = st.form_submit_button("Prédire la consommation", type="primary")

    if not submitted:
        st.info("Renseignez un profil puis lancez la prédiction.")
        return

    input_row = make_prediction_row(
        household_size=household_size,
        avg_temperature=avg_temperature,
        has_ac=has_ac,
    )
    model = st.session_state["training_result"]["model"]
    prediction = float(model.predict(input_row)[0])
    dataset_mean = float(prepared_df[TARGET_COLUMN].mean())

    col_result, col_features = st.columns([0.9, 1.1])
    with col_result:
        render_metric_card("Consommation prédite", f"{prediction:.2f} kWh", "Estimation du modèle")
        render_prediction_gauge(prediction, prepared_df)
    with col_features:
        st.subheader("Variables envoyées au modèle")
        st.dataframe(input_row, use_container_width=True, hide_index=True)
        if prediction > dataset_mean * 1.15:
            message = "Ce foyer se situe au-dessus de la moyenne du dataset."
            color = "rose"
        elif prediction < dataset_mean * 0.85:
            message = "Ce foyer se situe en dessous de la moyenne du dataset."
            color = "blue"
        else:
            message = "Ce foyer est proche de la moyenne observée dans le dataset."
            color = "purple"
        render_insight(
            "Interprétation",
            f"{message} Moyenne de référence : {dataset_mean:.2f} kWh.",
            color,
        )


def render_footer() -> None:
    """Render a short footer."""

    st.markdown(
        """
        <div class="footer">
            Projet Machine Learning · Régression énergétique des foyers
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_navigation() -> tuple[str, Any | None]:
    """Render the premium top navigation and dataset import controls."""

    st.markdown(
        """
        <div class="top-bar">
            <div class="top-bar-row">
                <div class="brand-lockup">
                    <div class="brand-mark">ML</div>
                    <div>
                        <p class="brand-title">Energy ML</p>
                        <p class="brand-subtitle">Application de régression énergétique</p>
                    </div>
                </div>
                <div class="target-pill">Cible : Energy_Consumption_kWh</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation principale",
        NAVIGATION_PAGES,
        horizontal=True,
        label_visibility="collapsed",
    )

    with st.expander("Importer un CSV personnalisé", expanded=False):
        uploaded_file = st.file_uploader(
            "Déposer ou sélectionner un fichier CSV",
            type=["csv"],
        )
        st.caption(
            "Sans import, l'application utilise automatiquement le dataset du projet."
        )

    return page, uploaded_file


def build_app() -> None:
    """Build the full Streamlit application."""

    st.set_page_config(
        page_title="Prédiction énergétique des foyers",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    page, uploaded_file = render_top_navigation()

    try:
        df = load_dataset(uploaded_file)
        missing_columns = validate_dataset(df)
        if missing_columns:
            st.error("Le dataset ne contient pas toutes les colonnes nécessaires.")
            st.write(", ".join(missing_columns))
            st.stop()
        prepared_df = prepare_features(df)
    except Exception as exc:
        st.error("Impossible de charger ou préparer les données.")
        st.exception(exc)
        st.stop()

    if page == "Accueil":
        render_home(df, prepared_df)
    elif page == "Données & EDA":
        render_data_page(df, prepared_df, uploaded_file)
    elif page == "Modélisation":
        render_model_page(prepared_df)
    elif page == "Prédiction":
        render_prediction_page(prepared_df)

    render_footer()


if __name__ == "__main__":
    build_app()
