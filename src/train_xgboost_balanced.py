import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ===============================
# PATHS
# ===============================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("🚀 Phase 2.2: Balancing Data with SMOTE and Retraining XGBoost")

# ===============================
# LOAD DATA
# ===============================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reduced.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reduced.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_reduced.csv")).values.ravel()

print(f"✅ Original training data: {X_train.shape}, Class counts: {np.bincount(y_train)}")

# ===============================
# APPLY SMOTE
# ===============================
print("\n🔄 Applying SMOTE balancing...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_bal.shape}, Class counts: {np.bincount(y_train_bal)}")

# ===============================
# BEST PARAMETERS (from tuning)
# ===============================
best_params = {
    'colsample_bytree': 0.7,
    'gamma': 0,
    'learning_rate': 0.03,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8
}

# ===============================
# TRAIN MODEL
# ===============================
print("\n⚙️ Training XGBoost on balanced data...")
xgb_bal = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
xgb_bal.fit(X_train_bal, y_train_bal)
print("✅ Balanced model trained successfully!")

# ===============================
# EVALUATION
# ===============================
print("\n📊 Evaluating model on original test data...")
y_pred = xgb_bal.predict(X_test)
y_proba = xgb_bal.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print(f"\n🎯 Balanced Model Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

with open(os.path.join(REPORT_DIR, "xgb_balanced_metrics.txt"), "w") as f:
    f.write("XGBoost Balanced Model Metrics\n")
    f.write("==============================\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=3))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix - XGBoost (Balanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgb_balanced_confusion_matrix.png"))
plt.close()
print("✅ Confusion matrix saved to reports/xgb_balanced_confusion_matrix.png")

# ===============================
# FEATURE IMPORTANCE
# ===============================
plt.figure(figsize=(8, 6))
plot_importance(xgb_bal, max_num_features=10, importance_type="gain")
plt.title("Top 10 Feature Importance (XGBoost - Balanced)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgb_balanced_feature_importance.png"))
plt.close()
print("✅ Feature importance plot saved to reports/xgb_balanced_feature_importance.png")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(xgb_bal, os.path.join(MODEL_DIR, "xgboost_balanced.joblib"))
print("💾 Final balanced model saved → models/xgboost_balanced.joblib")

print("\n🎉 Phase 2.2 Completed Successfully — Balanced XGBoost Model Ready!")
