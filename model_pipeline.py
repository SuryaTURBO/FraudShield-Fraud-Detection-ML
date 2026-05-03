"""
FraudShield - Modeling and validation pipeline.

This script trains and validates a credit card fraud detection system using the
explicit real-world feature schema requested for the FraudShield dashboard.

Required dataset:
    C:\\Users\\Suryansh\\OneDrive\\Desktop\\FraudShield\\credit_fraud_284k.csv

Deployment gate:
    The XGBoost model is exported only when:
    1. The dataset can be canonicalized to the exact expected feature schema.
    2. Holdout accuracy is at least 95%.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_DEPLOYMENT_ACCURACY = 0.95

PROJECT_DIR = Path(r"C:\Users\Suryansh\OneDrive\Desktop\FraudShield")
DATASET_PATH = PROJECT_DIR / "credit_fraud_284k.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "fraudshield_xgboost_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "fraudshield_robust_scaler.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "fraudshield_feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "fraudshield_model_metrics.json"
VALIDATION_REPORT_PATH = ARTIFACTS_DIR / "fraudshield_validation_report.json"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "age",
    "Amount",
    "account_balance",
    "num_transactions_today",
    "is_foreign_transaction",
    "transaction_hour",
    "prev_fraud_flag",
    "merchant_distance_km",
    "merchant_risk_score",
    "Time",
]

CONTINUOUS_FEATURES = [
    "age",
    "Amount",
    "account_balance",
    "num_transactions_today",
    "transaction_hour",
    "merchant_distance_km",
    "merchant_risk_score",
    "Time",
]

BINARY_FEATURES = [
    "is_foreign_transaction",
    "prev_fraud_flag",
]

COLUMN_ALIASES = {
    "Amount": ["Amount", "transaction_amount", "Transfer Amount", "transfer_amount"],
    "account_balance": [
        "account_balance",
        "Ledger Amount",
        "ledger_amount",
        "ledger_balance",
    ],
    "Class": ["Class", "is_fraud", "fraud", "target"],
}


@dataclass
class ModelResult:
    """Holdout evaluation metrics for one model."""

    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    confusion_matrix: list[list[int]]


@dataclass
class ValidationReport:
    """Dataset and deployment validation result."""

    dataset_path: str
    rows: int
    original_columns: list[str]
    canonical_feature_columns: list[str]
    schema_valid: bool
    schema_errors: list[str]
    xgboost_accuracy: float
    accuracy_threshold: float
    deployment_approved: bool


def ensure_artifact_dir() -> None:
    """Create artifact directory if needed."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Find a column by canonical name or configured aliases."""
    candidates = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    normalized_lookup = {column.strip().lower(): column for column in df.columns}

    for candidate in candidates:
        match = normalized_lookup.get(candidate.strip().lower())
        if match:
            return match

    return None


def load_raw_dataset() -> pd.DataFrame:
    """Load the exact required FraudShield dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Required dataset not found. Expected file at: "
            f"{DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


def canonicalize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert supported raw column names into the exact FraudShield schema.

    The current dataset contains transaction_hour but no raw Time. To preserve
    the requested 10-feature deployment schema, Time is derived as seconds from
    the hour of day: Time = transaction_hour * 3600.
    """
    result = pd.DataFrame()
    errors: list[str] = []

    direct_columns = [
        "age",
        "num_transactions_today",
        "is_foreign_transaction",
        "transaction_hour",
        "prev_fraud_flag",
        "merchant_distance_km",
        "merchant_risk_score",
    ]

    for column in direct_columns:
        if column in df.columns:
            result[column] = df[column]
        else:
            errors.append(f"Missing required feature column: {column}")

    for canonical_name in ["Amount", "account_balance", TARGET_COLUMN]:
        source_column = find_column(df, canonical_name)
        if source_column is None:
            errors.append(
                f"Missing required column or alias for: {canonical_name}"
            )
        else:
            result[canonical_name] = df[source_column]

    if "Time" in df.columns:
        result["Time"] = df["Time"]
    elif "transaction_hour" in result.columns:
        result["Time"] = pd.to_numeric(
            result["transaction_hour"],
            errors="coerce",
        ) * 3600.0
    else:
        errors.append("Missing Time and transaction_hour, cannot derive Time.")

    if errors:
        return result, errors

    ordered = result[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")

    ordered.replace([np.inf, -np.inf], np.nan, inplace=True)
    ordered[FEATURE_COLUMNS] = ordered[FEATURE_COLUMNS].fillna(
        ordered[FEATURE_COLUMNS].median(numeric_only=True)
    )
    ordered[TARGET_COLUMN] = ordered[TARGET_COLUMN].fillna(0).astype(int)

    for binary_column in BINARY_FEATURES:
        ordered[binary_column] = ordered[binary_column].round().clip(0, 1).astype(int)

    ordered["transaction_hour"] = ordered["transaction_hour"].round().clip(0, 23)

    if ordered[TARGET_COLUMN].nunique() != 2:
        errors.append("Target column must contain both classes: 0 and 1.")

    return ordered, errors


def validate_schema(df: pd.DataFrame, schema_errors: list[str]) -> tuple[bool, list[str]]:
    """Validate that the canonicalized dataset exactly matches model schema."""
    errors = list(schema_errors)
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]

    if feature_columns != FEATURE_COLUMNS:
        errors.append(
            "Canonical feature schema mismatch. Expected "
            f"{FEATURE_COLUMNS}, found {feature_columns}."
        )

    missing_values = int(df[FEATURE_COLUMNS].isna().sum().sum()) if not df.empty else 0
    if missing_values:
        errors.append(f"Feature matrix contains {missing_values} missing values.")

    return len(errors) == 0, errors


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """Fit scaler for continuous numeric features."""
    scaler = RobustScaler()
    scaler.fit(X_train[CONTINUOUS_FEATURES])
    return scaler


def transform_features(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Apply fitted scaler while keeping binary flags unchanged."""
    transformed = X[FEATURE_COLUMNS].copy()
    transformed[CONTINUOUS_FEATURES] = scaler.transform(
        transformed[CONTINUOUS_FEATURES]
    )
    return transformed


def build_models() -> dict[str, Any]:
    """Build all benchmark models."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=18,
            min_samples_leaf=2,
            n_estimators=220,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            colsample_bytree=0.9,
            eval_metric="logloss",
            gamma=0.05,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            n_estimators=350,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            reg_lambda=1.2,
            subsample=0.95,
            tree_method="hist",
        ),
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    """Evaluate a fitted model on holdout data."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probability_scores = model.predict_proba(X_test)[:, 1]
    else:
        probability_scores = predictions

    return ModelResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, probability_scores)),
        confusion_matrix=confusion_matrix(y_test, predictions).astype(int).tolist(),
    )


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict[str, ModelResult], Any, RobustScaler]:
    """Train benchmark models with SMOTE applied only to the training set."""
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = fit_scaler(X_train)
    X_train_scaled = transform_features(X_train, scaler)
    X_test_scaled = transform_features(X_test, scaler)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled,
        y_train,
    )

    results: dict[str, ModelResult] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        fitted_model = clone(model)
        fitted_model.fit(X_train_balanced, y_train_balanced)
        results[model_name] = evaluate_model(fitted_model, X_test_scaled, y_test)
        fitted_models[model_name] = fitted_model

        model_result = results[model_name]
        print(
            f"{model_name}: "
            f"accuracy={model_result.accuracy:.4f}, "
            f"precision={model_result.precision:.4f}, "
            f"recall={model_result.recall:.4f}, "
            f"roc_auc={model_result.roc_auc:.4f}"
        )

    return results, fitted_models["XGBoost"], scaler


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON data with indentation."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_deployment_artifacts(
    model: Any,
    scaler: RobustScaler,
    metrics: dict[str, ModelResult],
    validation_report: ValidationReport,
) -> None:
    """Save model, scaler, feature columns, metrics, and validation report."""
    ensure_artifact_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    save_json(
        {
            "feature_columns": FEATURE_COLUMNS,
            "continuous_features": CONTINUOUS_FEATURES,
            "binary_features": BINARY_FEATURES,
            "target_column": TARGET_COLUMN,
        },
        FEATURE_COLUMNS_PATH,
    )
    save_json({name: asdict(result) for name, result in metrics.items()}, METRICS_PATH)
    save_json(asdict(validation_report), VALIDATION_REPORT_PATH)


def run_pipeline() -> None:
    """Execute the complete FraudShield training and deployment gate."""
    ensure_artifact_dir()

    print(f"Loading dataset: {DATASET_PATH}")
    raw_df = load_raw_dataset()
    canonical_df, schema_errors = canonicalize_dataset(raw_df)
    schema_valid, validation_errors = validate_schema(canonical_df, schema_errors)

    if not schema_valid:
        report = ValidationReport(
            dataset_path=str(DATASET_PATH),
            rows=int(len(raw_df)),
            original_columns=list(raw_df.columns),
            canonical_feature_columns=[
                column for column in canonical_df.columns if column != TARGET_COLUMN
            ],
            schema_valid=False,
            schema_errors=validation_errors,
            xgboost_accuracy=0.0,
            accuracy_threshold=MIN_DEPLOYMENT_ACCURACY,
            deployment_approved=False,
        )
        save_json(asdict(report), VALIDATION_REPORT_PATH)
        raise ValueError(
            "Dataset schema validation failed. Details saved to "
            f"{VALIDATION_REPORT_PATH}"
        )

    metrics, xgboost_model, scaler = train_and_evaluate(canonical_df)
    xgboost_accuracy = metrics["XGBoost"].accuracy
    deployment_approved = xgboost_accuracy >= MIN_DEPLOYMENT_ACCURACY

    report = ValidationReport(
        dataset_path=str(DATASET_PATH),
        rows=int(len(raw_df)),
        original_columns=list(raw_df.columns),
        canonical_feature_columns=FEATURE_COLUMNS,
        schema_valid=True,
        schema_errors=[],
        xgboost_accuracy=xgboost_accuracy,
        accuracy_threshold=MIN_DEPLOYMENT_ACCURACY,
        deployment_approved=deployment_approved,
    )

    save_json({name: asdict(result) for name, result in metrics.items()}, METRICS_PATH)
    save_json(asdict(report), VALIDATION_REPORT_PATH)

    if not deployment_approved:
        raise RuntimeError(
            "Deployment gate failed: XGBoost accuracy "
            f"{xgboost_accuracy:.2%} is below the required "
            f"{MIN_DEPLOYMENT_ACCURACY:.2%}. Metrics saved to {METRICS_PATH}."
        )

    save_deployment_artifacts(xgboost_model, scaler, metrics, report)
    print("Deployment gate passed.")
    print(f"Saved XGBoost model: {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    run_pipeline()
