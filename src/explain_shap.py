"""
Phase 2.5 - SHAP Explainability for Balanced XGBoost Model
This script generates global SHAP plots (bar importance + summary)
and skips the local force plot to avoid visualization issues.
"""

import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 📁 Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_balanced.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "X_test_reduced.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

print("🚀 Phase 2.5: SHAP Explainability for Balanced XGBoost Model")

# -----------------------------
# 📦 Load model & test data
# -----------------------------
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

X_test = pd.read_csv(DATA_PATH)
print(f"✅ Loaded test data: {X_test.shape}")

# -----------------------------
# ⚙️ Compute SHAP values (fast mode)
# -----------------------------
print("\n⚙️ Computing SHAP values (TreeExplainer, fast mode)...")

# Use a sample of 2000 rows for faster computation
X_sample = X_test.sample(n=2000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

print(f"✅ SHAP values computed successfully! Shape: {shap_values.values.shape}")

# -----------------------------
# 📊 Global Explainability Plots
# -----------------------------
print("\n📊 Generating global SHAP plots...")

# 1️⃣ Bar Plot (Feature Importance)
plt.title("Top Features by Mean |SHAP| Value")
shap.summary_plot(
    shap_values.values,
    X_sample,
    plot_type="bar",
    show=False
)
plt.tight_layout()
bar_path = os.path.join(REPORT_DIR, "shap_bar_importance.png")
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"✅ Saved: {bar_path}")

# 2️⃣ Summary Plot (Feature Impact Distribution)
shap.summary_plot(
    shap_values.values,
    X_sample,
    show=False
)
plt.tight_layout()
summary_path = os.path.join(REPORT_DIR, "shap_summary_plot.png")
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"✅ Saved: {summary_path}")

# -----------------------------
# 🧾 Completion Message
# -----------------------------
print("\n🎉 Phase 2.5 Completed Successfully — SHAP Explainability Done!")
print(f"📁 Reports saved in: {REPORT_DIR}")
