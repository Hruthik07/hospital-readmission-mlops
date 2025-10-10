"""
🚀 Phase 3.3 (Final Fixed Version) — Manually log reduced-feature XGBoost model to MLflow
This script:
  - Loads the reduced 10-feature test data
  - Evaluates multiclass metrics automatically
  - Logs model + metrics to MLflow Tracking Server
"""

import mlflow
import mlflow.xgboost
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("🚀 Logging reduced-feature XGBoost model to MLflow...")

# ================================
# 🔹 Paths
# ================================
MODEL_PATH = "models/xgboost_balanced.joblib"
X_TEST_PATH = "data/processed/X_test_reduced.csv"
Y_TEST_PATH = "data/processed/y_test_reduced.csv"

# ================================
# 🔹 Load Model and Data
# ================================
model = load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # flatten if it's a single column dataframe

# ================================
# 🔹 Make Predictions
# ================================
y_pred = model.predict(X_test)

# Automatically detect if target is multiclass or binary
num_classes = len(set(y_test))
if num_classes > 2:
    avg_type = "weighted"
    roc_auc = roc_auc_score(
        y_test, model.predict_proba(X_test), multi_class="ovr"
    )
else:
    avg_type = "binary"
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Evaluate metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=avg_type)

print(f"✅ Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")

# ================================
# 🔹 Log to MLflow
# ================================
mlflow.set_experiment("Hospital_Readmission_Tracking")

with mlflow.start_run(run_name="Manual_Logged_Reduced_XGBoost") as run:
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.xgboost.log_model(model, artifact_path="model")

    print(f"\n✅ Model logged successfully under run_id={run.info.run_id}")

# ================================
# 🔹 Completion Message
# ================================
print("\n🎯 Open MLflow UI → http://127.0.0.1:5000 to verify the logged model.")
