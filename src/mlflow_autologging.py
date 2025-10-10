"""
Phase 3.2 - MLflow Autologging and Multi-Run Comparison
Tracks baseline, tuned, and balanced XGBoost models automatically.
"""

import os
import mlflow
import mlflow.xgboost
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("🚀 Phase 3.2: MLflow Autologging for Multiple Models")

# Load data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reduced.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reduced.csv")).squeeze()
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

mlflow.set_experiment("Hospital_Readmission_Tracking")

# Enable MLflow autologging
mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

# Run 1️⃣: Baseline Model
with mlflow.start_run(run_name="Baseline_XGBoost"):
    baseline_model = XGBClassifier(random_state=42, eval_metric="mlogloss")
    baseline_model.fit(X_train, y_train)
    preds = baseline_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    roc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(preds), average="weighted")
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc})
    print(f"✅ Baseline → Accuracy={acc:.4f} | F1={f1:.4f} | ROC={roc:.4f}")

# Run 2️⃣: Tuned Model (load existing)
with mlflow.start_run(run_name="Tuned_XGBoost"):
    tuned_model = load(os.path.join(MODEL_DIR, "xgboost_tuned.joblib"))
    preds = tuned_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    roc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(preds), average="weighted")
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc})
    print(f"✅ Tuned → Accuracy={acc:.4f} | F1={f1:.4f} | ROC={roc:.4f}")

# Run 3️⃣: Balanced Model (load existing)
with mlflow.start_run(run_name="Balanced_XGBoost"):
    balanced_model = load(os.path.join(MODEL_DIR, "xgboost_balanced.joblib"))
    preds = balanced_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    roc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(preds), average="weighted")
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc})
    print(f"✅ Balanced → Accuracy={acc:.4f} | F1={f1:.4f} | ROC={roc:.4f}")

print("\n🎯 Phase 3.2 completed — all model runs tracked in MLflow UI.")
print("💾 View at: http://127.0.0.1:5000 → Experiment: Hospital_Readmission_Tracking")
