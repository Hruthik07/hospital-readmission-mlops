"""
Data Preprocessing Pipeline for Hospital Readmission Project.

This module handles data loading, cleaning, encoding, scaling, and train-test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# ==============================
# PATH SETUP
# ==============================
RAW_DATA_PATH = "data/raw/hospital_readmission.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("🚀 Starting preprocessing...")

# ==============================
# LOAD RAW DATA
# ==============================
df = pd.read_csv(RAW_DATA_PATH)
print(f"✅ Data loaded successfully. Shape: {df.shape}")

# ==============================
# HANDLE MISSING VALUES
# ==============================
# Drop rows with missing target
if "readmitted" not in df.columns:
    raise ValueError("❌ Target column 'readmitted' not found in dataset!")

df = df.dropna(subset=["readmitted"])
# Fill other NaNs with 0 for simplicity (or you can use median/imputation)
df = df.fillna(0)
print("✅ Missing values handled.")

# ==============================
# ENCODE CATEGORICAL COLUMNS
# ==============================
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure consistent string type
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(f"✅ Encoded {len(categorical_cols)} categorical columns safely.")

# ==============================
# SPLIT FEATURES & TARGET
# ==============================
X = df.drop("readmitted", axis=1)
y = df["readmitted"]
print("✅ Split into features and target.")

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Scaling complete.")

# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✅ Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==============================
# SAVE PROCESSED DATA WITH COLUMN NAMES
# ==============================
# Convert scaled arrays back to DataFrames with original column names
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
y_train_df = pd.DataFrame(y_train, columns=["readmitted"])
y_test_df = pd.DataFrame(y_test, columns=["readmitted"])

# Save to CSVs
X_train_df.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test_df.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train_df.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test_df.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

# Save transformers
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))
joblib.dump(label_encoders, os.path.join(PROCESSED_DIR, "label_encoders.joblib"))

print("\n🎯 Preprocessing complete!")
print("✅ Saved files:")
print(f"   - {os.path.join(PROCESSED_DIR, 'X_train.csv')}")
print(f"   - {os.path.join(PROCESSED_DIR, 'X_test.csv')}")
print(f"   - {os.path.join(PROCESSED_DIR, 'y_train.csv')}")
print(f"   - {os.path.join(PROCESSED_DIR, 'y_test.csv')}")
print(f"   - {os.path.join(PROCESSED_DIR, 'scaler.joblib')}")
print(f"   - {os.path.join(PROCESSED_DIR, 'label_encoders.joblib')}")
