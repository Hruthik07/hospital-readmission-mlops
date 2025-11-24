
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
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

print("🚀 Starting Phase 2.1: Hyperparameter Tuning for XGBoost")

# ===============================
# LOAD DATA
# ===============================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_reduced.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_reduced.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_reduced.csv")).values.ravel()

print(f"✅ Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

# ===============================
# GRID SEARCH
# ===============================
param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [5, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
}

xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="roc_auc_ovr",
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("🔍 Running GridSearchCV (this may take several minutes)...")
grid.fit(X_train, y_train)

best_params = grid.best_params_
print("\n🏆 Best Parameters Found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

with open(os.path.join(REPORT_DIR, "xgb_best_params.txt"), "w") as f:
    f.write("Best Hyperparameters for XGBoost:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")

# ===============================
# TRAIN FINAL MODEL
# ===============================
print("\n⚙️ Training final model with best parameters...")
xgb_tuned = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
xgb_tuned.fit(X_train, y_train)
print("✅ Tuned model trained successfully!")

# ===============================
# EVALUATE MODEL
# ===============================
print("\n📊 Evaluating tuned model...")
y_pred = xgb_tuned.predict(X_test)
y_proba = xgb_tuned.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print("\n🎯 Tuned Model Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

with open(os.path.join(REPORT_DIR, "xgb_tuned_metrics.txt"), "w") as f:
    f.write("Tuned XGBoost Model Metrics\n")
    f.write("===========================\n")
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - XGBoost (Tuned)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgb_tuned_confusion_matrix.png"))
plt.close()
print("✅ Confusion matrix saved to reports/xgb_tuned_confusion_matrix.png")

# ===============================
# FEATURE IMPORTANCE
# ===============================
plt.figure(figsize=(8, 6))
plot_importance(xgb_tuned, max_num_features=10, importance_type="gain")
plt.title("Top 10 Feature Importance (XGBoost - Tuned)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "xgb_tuned_feature_importance.png"))
plt.close()
print("✅ Feature importance plot saved to reports/xgb_tuned_feature_importance.png")

# ===============================
# SAVE FINAL MODEL
# ===============================
joblib.dump(xgb_tuned, os.path.join(MODEL_DIR, "xgboost_tuned.joblib"))
print("💾 Final tuned model saved → models/xgboost_tuned.joblib")

print("\n🎉 Phase 2.1 Completed Successfully — Tuned XGBoost Model Ready!")
