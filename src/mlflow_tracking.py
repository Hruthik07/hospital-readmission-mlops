"""
Phase 3 - MLflow Experiment Tracking
Tracks model parameters, metrics, and artifacts for comparison.
"""

import os
import mlflow
import mlflow.sklearn
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

# -----------------------------
# 📁 Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_balanced.joblib")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# 🧠 Load model & test data
# -----------------------------
print("🚀 Phase 3: MLflow Tracking for Hospital Readmission Project")

model = load(MODEL_PATH)
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"✅ Loaded model and test data: X_test={X_test.shape}, y_test={y_test.shape}")

# -----------------------------
# ⚙️ Predictions & metrics
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average="weighted")

print("\n📊 Evaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# -----------------------------
# 🧾 MLflow Tracking
# -----------------------------
mlflow.set_experiment("Hospital_Readmission_Tracking")

with mlflow.start_run(run_name="XGBoost_Balanced_Model"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("phase", "balanced + tuned")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log SHAP plots as artifacts
    mlflow.log_artifact(os.path.join(REPORT_DIR, "shap_bar_importance.png"))
    mlflow.log_artifact(os.path.join(REPORT_DIR, "shap_summary_plot.png"))

print("\n✅ MLflow logging complete!")
print("💾 Run 'mlflow ui' and open http://127.0.0.1:5000 to explore runs.")
