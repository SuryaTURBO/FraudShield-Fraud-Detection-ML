"""
FraudShield - Streamlit dashboard.

Run from Command Prompt:
    cd /d "C:\\Users\\Suryansh\\OneDrive\\Desktop\\FraudShield"
    .\\.venv\\Scripts\\activate.bat
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model_pipeline import (
    CONTINUOUS_FEATURES,
    DATASET_PATH,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    VALIDATION_REPORT_PATH,
    canonicalize_dataset,
    load_raw_dataset,
    transform_features,
)


PROJECT_DIR = Path(r"C:\Users\Suryansh\OneDrive\Desktop\FraudShield")
APP_DATASET_NAME = "credit_fraud_284k.csv"
APP_DATASET_PATH = PROJECT_DIR / APP_DATASET_NAME
DEFAULT_TRANSACTION_HOUR = 12
DEFAULT_IS_FOREIGN_TRANSACTION = 0
DEFAULT_PREV_FRAUD_FLAG = 0
DEFAULT_MERCHANT_DISTANCE_KM = 0.0
AGE_RISK_MIN = 16
AGE_RISK_MAX = 55
AGE_RISK_BASE_BOOST = 0.55
AGE_RISK_STEP_BOOST = 0.02
AGE_RISK_MAX_BOOST = 0.95
HIDDEN_LIVE_FEATURES = {
    "is_foreign_transaction",
    "transaction_hour",
    "prev_fraud_flag",
    "merchant_distance_km",
    "Time",
}

if DATASET_PATH != APP_DATASET_PATH:
    raise RuntimeError(
        "FraudShield app is configured only for "
        f"{APP_DATASET_NAME}; model_pipeline DATASET_PATH is {DATASET_PATH}"
    )


st.set_page_config(
    page_title="FraudShield",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background: #0b1120;
            color: #e5e7eb;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }
        .fraud-title {
            color: #ffffff;
            font-size: 2.35rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .fraud-subtitle {
            color: #cbd5e1;
            font-size: 1.02rem;
            text-align: center;
            margin-bottom: 1.6rem;
        }
        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        .sidebar-footer {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.55;
            padding-top: 2.25rem;
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
            color: #ffffff !important;
        }
        div[data-testid="stForm"] {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
        }
        .stDataFrame {
            border: 1px solid #334155;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler() -> tuple[Any | None, Any | None]:
    """Load deployment model and scaler."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None

    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def load_json_file(path: Path) -> dict[str, Any]:
    """Load JSON file if present."""
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dashboard_dataset() -> pd.DataFrame:
    """Load and canonicalize the 284k transaction dataset for dashboard visuals."""
    if DATASET_PATH.name != APP_DATASET_NAME:
        raise ValueError(f"Dashboard is configured only for {APP_DATASET_NAME}.")

    raw_df = load_raw_dataset()
    canonical_df, schema_errors = canonicalize_dataset(raw_df)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    return canonical_df


def apply_dark_chart_style(ax: plt.Axes, title: str | None = None) -> None:
    """Apply consistent dark dashboard styling to a Matplotlib axis."""
    ax.set_facecolor("#0f172a")
    ax.figure.set_facecolor("#0b1120")
    ax.tick_params(colors="#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    if title:
        ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold")
    if ax.legend_:
        ax.legend_.get_frame().set_facecolor("#111827")
        ax.legend_.get_frame().set_edgecolor("#334155")
        for text in ax.legend_.get_texts():
            text.set_color("#e5e7eb")


def render_class_distribution(df: pd.DataFrame) -> None:
    """Plot legitimate vs fraud transaction count."""
    counts = (
        df[TARGET_COLUMN]
        .map({0: "Legitimate", 1: "Fraud"})
        .value_counts()
        .reindex(["Legitimate", "Fraud"])
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax,
        palette=["#22c55e", "#ef4444"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Transactions")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{int(value):,}", ha="center", va="bottom", color="#fff")
    apply_dark_chart_style(ax, "Class Distribution")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_hourly_activity(df: pd.DataFrame) -> None:
    """Plot transaction volume by hour and class."""
    hourly = (
        df.assign(ClassLabel=df[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraud"}))
        .groupby(["transaction_hour", "ClassLabel"])
        .size()
        .reset_index(name="Transactions")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=hourly,
        x="transaction_hour",
        y="Transactions",
        hue="ClassLabel",
        marker="o",
        palette={"Legitimate": "#38bdf8", "Fraud": "#fb7185"},
        ax=ax,
    )
    ax.set_xlabel("Transaction Hour")
    ax.set_ylabel("Transactions")
    ax.set_xticks(range(0, 24, 2))
    apply_dark_chart_style(ax, "Transaction Volume by Hour")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_fraud_rate_by_hour(df: pd.DataFrame) -> None:
    """Plot fraud rate by transaction hour."""
    hourly_rate = (
        df.groupby("transaction_hour")[TARGET_COLUMN]
        .mean()
        .mul(100)
        .reset_index(name="Fraud Rate")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=hourly_rate,
        x="transaction_hour",
        y="Fraud Rate",
        marker="o",
        color="#f59e0b",
        ax=ax,
    )
    ax.fill_between(
        hourly_rate["transaction_hour"],
        hourly_rate["Fraud Rate"],
        color="#f59e0b",
        alpha=0.18,
    )
    ax.set_xlabel("Transaction Hour")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xticks(range(0, 24, 2))
    apply_dark_chart_style(ax, "Fraud Rate by Hour")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_amount_distribution(df: pd.DataFrame) -> None:
    """Plot transaction amount distribution by class."""
    sample = df.copy()
    sample["ClassLabel"] = sample[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraud"})
    upper_amount = sample["Amount"].quantile(0.98)
    sample = sample[sample["Amount"] <= upper_amount]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=sample,
        x="Amount",
        hue="ClassLabel",
        bins=45,
        element="step",
        stat="density",
        common_norm=False,
        palette={"Legitimate": "#38bdf8", "Fraud": "#fb7185"},
        ax=ax,
    )
    ax.set_xlabel("Amount")
    ax.set_ylabel("Density")
    apply_dark_chart_style(ax, "Transaction Amount Distribution")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_risk_boxplot(df: pd.DataFrame) -> None:
    """Plot merchant risk score spread by class."""
    plot_df = df.copy()
    plot_df["ClassLabel"] = plot_df[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=plot_df,
        x="ClassLabel",
        y="merchant_risk_score",
        ax=ax,
        palette=["#22c55e", "#ef4444"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Merchant Risk Score")
    apply_dark_chart_style(ax, "Merchant Risk Score by Class")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot feature correlation heatmap including target."""
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        linecolor="#1e293b",
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    apply_dark_chart_style(ax, "Feature Correlation Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="#e5e7eb")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#e5e7eb")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="#e5e7eb")
    plt.setp(cbar.ax.get_yticklabels(), color="#e5e7eb")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Convert metrics JSON into a model comparison DataFrame."""
    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy": values.get("accuracy", 0.0),
                "Precision": values.get("precision", 0.0),
                "Recall": values.get("recall", 0.0),
                "ROC AUC": values.get("roc_auc", 0.0),
            }
            for model_name, values in metrics.items()
        ]
    )


def render_model_metric_bars(metrics: dict[str, Any]) -> None:
    """Plot model metrics as grouped bars."""
    benchmark = metrics_dataframe(metrics)
    if benchmark.empty:
        return
    long_df = benchmark.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "ROC AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=long_df,
        x="Model",
        y="Score",
        hue="Metric",
        ax=ax,
        palette=["#38bdf8", "#22c55e", "#f59e0b", "#a78bfa"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    apply_dark_chart_style(ax, "Model Benchmark Metrics")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_confusion_matrices(metrics: dict[str, Any]) -> None:
    """Render confusion matrices for all benchmark models."""
    st.markdown("#### Confusion Matrices")
    columns = st.columns(len(metrics))
    for column, (model_name, values) in zip(columns, metrics.items(), strict=False):
        matrix = np.array(values.get("confusion_matrix", [[0, 0], [0, 0]]))
        with column:
            fig, ax = plt.subplots(figsize=(4.3, 3.7))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            apply_dark_chart_style(ax, model_name)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_feature_importance() -> None:
    """Render deployed XGBoost feature importance."""
    if not MODEL_PATH.exists():
        st.info("Feature importance is available after model deployment.")
        return
    model = joblib.load(MODEL_PATH)
    importances = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=importances,
        x="Importance",
        y="Feature",
        color="#38bdf8",
        ax=ax,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    apply_dark_chart_style(ax, "XGBoost Feature Importance")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def display_header() -> None:
    """Render dashboard header."""
    st.markdown(
        '<div class="fraud-title">FraudShield: ML Fraud Detection System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fraud-subtitle">'
        "Manual credit card fraud analysis powered by XGBoost, SMOTE, "
        "and robust transaction-risk features."
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render sidebar navigation and authorship."""
    st.sidebar.title("FraudShield")
    view = st.sidebar.radio(
        "Navigation",
        ["Project Overview", "Model Validation", "Live Fraud Detection"],
    )
    st.sidebar.markdown(
        '<div class="sidebar-footer">'
        "Mini Project by<br>"
        "Suryansh Mahatha<br>"
        "Thrishal Shetty<br>"
        "Pranay Pratap Singh"
        "</div>",
        unsafe_allow_html=True,
    )
    return view


def artifacts_ready() -> bool:
    """Check that required deployment artifacts exist."""
    missing = [
        path
        for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH]
        if not path.exists()
    ]

    if not missing:
        return True

    st.error(
        "Deployment artifacts are missing. Run `python model_pipeline.py` "
        "successfully before using Live Fraud Detection."
    )
    with st.expander("Missing artifact files"):
        for path in missing:
            st.write(path)
    return False


def render_project_overview() -> None:
    """Display project summary and expected feature contract."""
    st.subheader("Project Overview")

    validation_report = load_json_file(VALIDATION_REPORT_PATH)
    metrics = load_json_file(METRICS_PATH)
    report_dataset = (
        Path(validation_report.get("dataset_path", "")).name
        if validation_report
        else ""
    )
    if report_dataset and report_dataset != APP_DATASET_NAME:
        st.warning(
            f"Validation report is for {report_dataset}. "
            f"Run `python model_pipeline.py` for {APP_DATASET_NAME}."
        )

    try:
        df = load_dashboard_dataset()
    except (FileNotFoundError, ValueError) as exc:
        df = pd.DataFrame()
        st.warning(f"Dataset visualizations are unavailable: {exc}")

    rows = validation_report.get("rows", 0)
    deployment_approved = validation_report.get("deployment_approved", False)
    xgb_accuracy = validation_report.get("xgboost_accuracy", 0.0)

    fraud_rate = df[TARGET_COLUMN].mean() if not df.empty else 0.0
    average_amount = df["Amount"].mean() if not df.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", APP_DATASET_NAME)
    col2.metric("Dataset Rows", f"{rows:,}" if rows else "Not trained")
    col3.metric("XGBoost Accuracy", f"{xgb_accuracy:.2%}")
    col4.metric("Fraud Rate", f"{fraud_rate:.2%}")
    st.metric("Deployment Gate", "Passed" if deployment_approved else "Pending")

    if not df.empty:
        st.markdown("#### Transaction Intelligence Visuals")
        tab1, tab2, tab3 = st.tabs(
            ["Volume & Timing", "Amount & Risk", "Feature Relationships"]
        )
        with tab1:
            left, right = st.columns(2)
            with left:
                render_class_distribution(df)
            with right:
                render_hourly_activity(df)
            render_fraud_rate_by_hour(df)
        with tab2:
            left, right = st.columns(2)
            with left:
                render_amount_distribution(df)
            with right:
                render_risk_boxplot(df)
            st.metric("Average Transaction Amount", f"{average_amount:,.2f}")
        with tab3:
            render_correlation_heatmap(df)


def render_model_validation() -> None:
    """Display validation report and model benchmark table."""
    st.subheader("Model Validation")

    validation_report = load_json_file(VALIDATION_REPORT_PATH)
    metrics = load_json_file(METRICS_PATH)

    if not validation_report:
        st.warning("No validation report found. Run `python model_pipeline.py` first.")
        return

    report_dataset = Path(validation_report.get("dataset_path", "")).name
    if report_dataset != APP_DATASET_NAME:
        st.error(
            f"This app only supports {APP_DATASET_NAME}, but the validation "
            f"report is for {report_dataset or 'unknown dataset'}."
        )
        return

    st.caption(f"Active dataset: {APP_DATASET_NAME}")

    if metrics:
        st.markdown("#### Latest Model Accuracy Summary")
        summary = metrics_dataframe(metrics)
        st.dataframe(
            summary.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "ROC AUC": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Schema Valid",
        "Yes" if validation_report.get("schema_valid") else "No",
    )
    col2.metric(
        "XGBoost Accuracy",
        f"{validation_report.get('xgboost_accuracy', 0.0):.2%}",
    )
    col3.metric(
        "Required Accuracy",
        f"{validation_report.get('accuracy_threshold', 0.95):.2%}",
    )

    if validation_report.get("schema_errors"):
        st.error("Schema validation issues were found.")
        for error in validation_report["schema_errors"]:
            st.write(f"- {error}")

    if metrics:
        st.markdown("#### Benchmark Visuals")
        render_model_metric_bars(metrics)
        render_confusion_matrices(metrics)
        render_feature_importance()


def initialize_history() -> None:
    """Initialize manual transaction history in session state."""
    if "manual_transaction_history" not in st.session_state:
        st.session_state.manual_transaction_history = []
    if "latest_manual_check_id" not in st.session_state:
        st.session_state.latest_manual_check_id = 0


def validate_manual_inputs(values: dict[str, float | int]) -> list[str]:
    """Validate manual form values before model inference."""
    errors: list[str] = []

    for feature in FEATURE_COLUMNS:
        value = values.get(feature)
        if value is None:
            errors.append(f"{feature} is required.")
        elif pd.isna(value):
            errors.append(f"{feature} cannot be blank.")

    if values["age"] <= 0:
        errors.append("age must be greater than 0.")
    if values["Amount"] < 0:
        errors.append("Amount cannot be negative.")
    if values["account_balance"] < 0:
        errors.append("account_balance cannot be negative.")
    if values["num_transactions_today"] < 0:
        errors.append("num_transactions_today cannot be negative.")
    if values["merchant_distance_km"] < 0:
        errors.append("merchant_distance_km cannot be negative.")
    if values["merchant_risk_score"] < 0:
        errors.append("merchant_risk_score cannot be negative.")
    if not 0 <= values["transaction_hour"] <= 23:
        errors.append("transaction_hour must be between 0 and 23.")
    if values["Time"] < 0:
        errors.append("Time cannot be negative.")

    return errors


def hour_label(hour: int) -> str:
    """Format an hour integer as user-friendly 12-hour AM/PM text."""
    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:00 {period}"


def parse_hour_label(label: str) -> int:
    """Convert a 12-hour label like '9:00 AM' into a 0-23 hour integer."""
    time_part, period = label.split()
    hour = int(time_part.split(":")[0])
    if period == "AM":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def calculate_merchant_risk_score(
    amount: float,
    account_balance: float,
    num_transactions_today: int,
    is_foreign_transaction: int,
    transaction_hour: int,
    prev_fraud_flag: int,
    merchant_distance_km: float,
) -> float:
    """Calculate a 0-1 merchant risk score from manual transaction signals."""
    amount_score = min(amount / 20000.0, 1.0) * 0.24
    balance_pressure = min(amount / max(account_balance, 1.0), 1.0) * 0.12
    velocity_score = min(num_transactions_today / 50.0, 1.0) * 0.12
    distance_score = min(merchant_distance_km / 5000.0, 1.0) * 0.18
    foreign_score = 0.14 if is_foreign_transaction else 0.0
    previous_fraud_score = 0.16 if prev_fraud_flag else 0.0
    night_score = 0.09 if transaction_hour <= 5 or transaction_hour >= 22 else 0.0
    base_score = 0.05

    score = (
        base_score
        + amount_score
        + balance_pressure
        + velocity_score
        + distance_score
        + foreign_score
        + previous_fraud_score
        + night_score
    )
    return round(min(score, 0.99), 4)


def manual_input_form() -> tuple[bool, dict[str, float | int]]:
    """Render manual transaction form and return submitted values."""
    with st.form("manual_fraud_detection_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
            amount = st.number_input(
                "Amount / Transfer Amount",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f",
            )

        with col2:
            account_balance = st.number_input(
                "Account Balance / Ledger Amount",
                min_value=0.0,
                value=25000.0,
                step=500.0,
                format="%.2f",
            )
            num_transactions_today = st.number_input(
                "Number of Transactions Today",
                min_value=0,
                value=5,
                step=1,
            )

        transaction_hour = DEFAULT_TRANSACTION_HOUR
        is_foreign_transaction = DEFAULT_IS_FOREIGN_TRANSACTION
        prev_fraud_flag = DEFAULT_PREV_FRAUD_FLAG
        merchant_distance_km = DEFAULT_MERCHANT_DISTANCE_KM
        merchant_risk_score = calculate_merchant_risk_score(
            amount=float(amount),
            account_balance=float(account_balance),
            num_transactions_today=int(num_transactions_today),
            is_foreign_transaction=is_foreign_transaction,
            transaction_hour=int(transaction_hour),
            prev_fraud_flag=prev_fraud_flag,
            merchant_distance_km=float(merchant_distance_km),
        )
        st.info(f"Merchant Risk Score: {merchant_risk_score:.2%}")

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    values: dict[str, float | int] = {
        "age": int(age),
        "Amount": float(amount),
        "account_balance": float(account_balance),
        "num_transactions_today": int(num_transactions_today),
        "is_foreign_transaction": is_foreign_transaction,
        "transaction_hour": int(transaction_hour),
        "prev_fraud_flag": prev_fraud_flag,
        "merchant_distance_km": float(merchant_distance_km),
        "merchant_risk_score": merchant_risk_score,
        "Time": float(transaction_hour * 3600),
    }
    return submitted, values


def render_prediction(prediction: int, probability: float) -> str:
    """Render prediction banner and return readable class label."""
    probability_text = f"{probability:.2%}"

    if prediction == 1:
        st.error(f"ALERT: FRAUD DETECTED - Fraud probability: {probability_text}")
        return "Fraud"

    st.success(f"LEGITIMATE TRANSACTION - Fraud probability: {probability_text}")
    return "Legitimate"


def apply_age_risk_adjustment(
    values: dict[str, float | int],
    prediction: int,
    probability: float,
) -> tuple[int, float]:
    """Apply a variable fraud-risk boost for ages outside the live-check range."""
    age = int(values["age"])
    if AGE_RISK_MIN <= age <= AGE_RISK_MAX:
        return prediction, probability

    if age < AGE_RISK_MIN:
        age_distance = AGE_RISK_MIN - age
    else:
        age_distance = age - AGE_RISK_MAX

    age_boost = min(
        AGE_RISK_BASE_BOOST + (age_distance * AGE_RISK_STEP_BOOST),
        AGE_RISK_MAX_BOOST,
    )
    adjusted_probability = probability + ((1.0 - probability) * age_boost)
    adjusted_prediction = 1 if adjusted_probability >= 0.5 else prediction
    return adjusted_prediction, adjusted_probability


def add_history_row(
    values: dict[str, float | int],
    predicted_class: str,
    probability: float,
) -> None:
    """Append one manual transaction result to session history."""
    st.session_state.latest_manual_check_id += 1
    for existing_row in st.session_state.manual_transaction_history:
        existing_row["Current"] = ""

    row = dict(values)
    row["Current"] = "Yes"
    row["Check"] = st.session_state.latest_manual_check_id
    row["Predicted Class"] = predicted_class
    row["Fraud Probability"] = probability
    st.session_state.manual_transaction_history.append(row)


def render_history() -> None:
    """Render manual transaction history table."""
    if not st.session_state.manual_transaction_history:
        return

    st.markdown("#### Transaction Details History")
    history = pd.DataFrame(st.session_state.manual_transaction_history)
    visible_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in HIDDEN_LIVE_FEATURES
    ]
    ordered_columns = [
        "Current",
        "Check",
        "Predicted Class",
        "Fraud Probability",
        *visible_feature_columns,
    ]
    history = history[ordered_columns].sort_values("Check", ascending=False)
    history = history.rename(
        columns={
            "account_balance": "Account Balance",
            "num_transactions_today": "Today Transaction Frequency",
            "merchant_risk_score": "Merchant Risk Score",
            "age": "Age",
            "Amount": "Amount",
        }
    )

    def highlight_recent(row: pd.Series) -> list[str]:
        if row["Current"] == "Yes":
            style = "background-color: #1d4ed8; color: #ffffff; font-weight: 700"
            return [style] * len(row)
        return ["color: #ffffff"] * len(row)

    st.dataframe(
        history.style.apply(highlight_recent, axis=1).format(
            {
                "Fraud Probability": "{:.2%}",
                "Merchant Risk Score": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_live_fraud_detection() -> None:
    """Render manual live fraud detection workflow."""
    st.subheader("Live Fraud Detection")
    initialize_history()

    if not artifacts_ready():
        render_history()
        return

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Unable to load model or scaler artifacts.")
        render_history()
        return

    st.write(
        "Enter transaction details manually, then analyze the transaction "
        "using the deployed XGBoost model."
    )

    submitted, values = manual_input_form()

    if submitted:
        errors = validate_manual_inputs(values)
        if errors:
            st.error("Please fix the following input issues before analysis:")
            for error in errors:
                st.write(f"- {error}")
        else:
            input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            scaled_input = transform_features(input_df, scaler)
            prediction = int(model.predict(scaled_input)[0])
            probability = float(model.predict_proba(scaled_input)[0][1])
            prediction, probability = apply_age_risk_adjustment(
                values,
                prediction,
                probability,
            )
            predicted_class = render_prediction(prediction, probability)
            add_history_row(values, predicted_class, probability)

    render_history()


def main() -> None:
    """Run Streamlit application."""
    display_header()
    selected_view = render_sidebar()

    if selected_view == "Project Overview":
        render_project_overview()
    elif selected_view == "Model Validation":
        render_model_validation()
    elif selected_view == "Live Fraud Detection":
        render_live_fraud_detection()


if __name__ == "__main__":
    main()
