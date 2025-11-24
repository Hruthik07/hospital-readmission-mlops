"""
Hospital Readmission MLOps Project - Configuration Module.

This module contains configuration constants used across the project.
"""

import os

# ============================================
# PROJECT PATHS
# ============================================
# Get the project root (parent directory of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = "Hospital_Readmission_Model"
MODEL_VERSION = 1
RANDOM_STATE = 42

# ============================================
# DATA CONFIGURATION
# ============================================
TEST_SIZE = 0.2
TARGET_COLUMN = "readmitted"

# ============================================
# MLFLOW CONFIGURATION
# ============================================
MLFLOW_EXPERIMENT_NAME = "Hospital_Readmission_Tracking"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# ============================================
# XGBOOST BEST PARAMETERS
# ============================================
XGBOOST_BEST_PARAMS = {
    'colsample_bytree': 0.7,
    'gamma': 0,
    'learning_rate': 0.03,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ============================================
# CLASS LABELS
# ============================================
CLASS_LABELS = {
    0: "No readmission",
    1: "Readmitted (>30 days)",
    2: "Readmitted (<30 days)"
}

# ============================================
# API CONFIGURATION
# ============================================
API_TITLE = "Hospital Readmission Prediction API"
API_DESCRIPTION = "Predicts the likelihood of hospital readmission using an MLflow-registered XGBoost model."
API_VERSION = "1.2"
