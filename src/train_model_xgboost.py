
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==============================
# PATH SETUP
# ==============================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("🚀 Starting Phase 2: Model Training (XGBoost)")

# ==============================
# LOAD DATA
# ==============================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reduced.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reduced.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_reduced.csv")).values.ravel()

print(f"✅ Data loaded successfully: X_train={X_train.shape}, X_test={X_test.shape}")

# ==============================
# TRAIN XGBOOST MODEL
# ==============================
print("\n⚙️ Training XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
print("✅ Model training completed successfully!")

# ==============================
# EVALUATION
# ==============================
print("\n📊 Evaluating model performance...")
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print(f"\n🎯 Model Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# ==============================
# SAVE METRICS REPORT
# ==============================
with open(os.path.join(REPORT_DIR, "xgboost_model_metrics.txt"), "w") as f:
    f.write("XGBoost Model Evaluation Metrics\n")
    f.write("=================================\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=3))

print("✅ Model metrics saved to reports/xgboost_model_metrics.txt")

# ==============================
# CONFUSION MATRIX PLOT
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost (Top 10 Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgboost_confusion_matrix.png"))
plt.close()
print("✅ Confusion matrix saved to reports/xgboost_confusion_matrix.png")

# ==============================
# FEATURE IMPORTANCE PLOT
# ==============================
plt.figure(figsize=(8, 6))
plot_importance(xgb_model, max_num_features=10, importance_type="gain", title="Top 10 Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgboost_feature_importance.png"))
plt.close()
print("✅ Feature importance plot saved to reports/xgboost_feature_importance.png")

# ==============================
# SAVE FINAL MODEL
# ==============================
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "final_xgboost_model.joblib"))
print(f"\n💾 Final model saved to models/final_xgboost_model.joblib")

print("\n🎉 Phase 2 Completed Successfully: XGBoost Model Trained and Saved!")
