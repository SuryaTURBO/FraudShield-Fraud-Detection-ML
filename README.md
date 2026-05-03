# FraudShield - ML Fraud Detection System

FraudShield is a complete mini project for credit card fraud detection using Python, SMOTE, Logistic Regression, Random Forest, XGBoost, and Streamlit.

## Dataset

The pipeline expects a file named exactly:

```text
credit_fraud_284k.csv
```

The code first checks the project folder. If the file is not there, it checks:

```text
C:\Users\Suryansh\Downloads\credit_fraud_284k.csv
```

Required columns:

```text
Time, Amount, V1, V2, ..., V28, Class
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Models

```powershell
python model_pipeline.py
```

This generates:

- EDA plots
- 5-fold KFold benchmark metrics
- Trained Logistic Regression, Random Forest, and XGBoost models
- Saved RobustScaler
- Holdout test set for live dashboard inference

Artifacts are saved in:

```text
artifacts/
```

## Run Dashboard

```powershell
streamlit run app.py
```

Dashboard views:

- The Data Explorer
- Model Benchmarking
- Live Fraud Detection

