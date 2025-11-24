import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# ==============================
# PATHS
# ==============================
DATA_DIR = "data/processed"
RESULTS_DIR = "reports"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("🚀 Starting Feature Selection Phase...")

# ==============================
# LOAD DATA
# ==============================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

feature_names = X_train.columns
print(f"✅ Loaded processed data: X_train={X_train.shape}, X_test={X_test.shape}")

# ==============================
# 1️⃣ ANOVA F-Test (Statistical Importance)
# ==============================
print("\n📊 Running ANOVA F-test for feature relevance...")
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, y_train)
anova_scores = selector.scores_

anova_df = pd.DataFrame({
    "Feature": feature_names,
    "ANOVA_Score": anova_scores
}).sort_values("ANOVA_Score", ascending=False)

print("✅ ANOVA F-test completed. Top 10 features:")
print(anova_df.head(10))

# ==============================
# 2️⃣ RandomForest Importance (Model-Based)
# ==============================
print("\n🌲 Training RandomForest for feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importances = rf.feature_importances_

rf_df = pd.DataFrame({
    "Feature": feature_names,
    "RF_Importance": rf_importances
}).sort_values("RF_Importance", ascending=False)

print("✅ RandomForest importance calculated. Top 10 features:")
print(rf_df.head(10))

# ==============================
# 3️⃣ Combine Both Methods
# ==============================
combined = pd.merge(anova_df, rf_df, on="Feature")
combined["Combined_Score"] = (
    0.5 * combined["ANOVA_Score"].rank(pct=True) +
    0.5 * combined["RF_Importance"].rank(pct=True)
)
combined = combined.sort_values("Combined_Score", ascending=False)
top10_features = combined.head(10)

print("\n🔥 Final Top 10 Features:")
print(top10_features[["Feature", "Combined_Score"]])

# ==============================
# 4️⃣ SAVE FEATURE RESULTS
# ==============================
top10_feature_names = top10_features["Feature"].tolist()

pd.DataFrame({"Top_Features": top10_feature_names}).to_csv(
    os.path.join(RESULTS_DIR, "top10_features.csv"), index=False
)
joblib.dump(top10_feature_names, os.path.join(DATA_DIR, "top10_features.joblib"))
print("\n✅ Top 10 feature names saved to:")
print(f"   → {os.path.join(RESULTS_DIR, 'top10_features.csv')}")
print(f"   → {os.path.join(DATA_DIR, 'top10_features.joblib')}")

# ==============================
# 5️⃣ SAVE REDUCED DATASETS (Train/Test, X & y)
# ==============================
print("\n💾 Saving reduced datasets...")

# Reduce feature space
X_train_reduced = X_train[top10_feature_names]
X_test_reduced = X_test[top10_feature_names]

# Load y values (already loaded above)
y_train_df = pd.DataFrame(y_train, columns=["readmitted"])
y_test_df = pd.DataFrame(y_test, columns=["readmitted"])

# Save reduced datasets
X_train_reduced.to_csv(os.path.join(DATA_DIR, "X_train_reduced.csv"), index=False)
X_test_reduced.to_csv(os.path.join(DATA_DIR, "X_test_reduced.csv"), index=False)
y_train_df.to_csv(os.path.join(DATA_DIR, "y_train_reduced.csv"), index=False)
y_test_df.to_csv(os.path.join(DATA_DIR, "y_test_reduced.csv"), index=False)

print("✅ Saved reduced datasets:")
print("   - X_train_reduced.csv")
print("   - X_test_reduced.csv")
print("   - y_train_reduced.csv")
print("   - y_test_reduced.csv")

# ==============================
# 6️⃣ SAVE FEATURE IMPORTANCE REPORT
# ==============================
combined.to_csv(os.path.join(RESULTS_DIR, "all_feature_importance.csv"), index=False)
print(f"📁 Full feature ranking saved to: {os.path.join(RESULTS_DIR, 'all_feature_importance.csv')}")

print("\n🎯 Feature Selection Phase Completed Successfully!")
