"""
FastAPI Deployment Module for Hospital Readmission Prediction.

This module provides a REST API for making hospital readmission predictions
using a trained XGBoost model loaded from MLflow Model Registry.
"""

import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd

# -------------------------------------------------
# 🚀 Initialize FastAPI Application
# -------------------------------------------------
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predicts the likelihood of hospital readmission using an MLflow-registered XGBoost model.",
    version="1.2"
)

# -------------------------------------------------
# 🔹 Load the Registered Model from MLflow Registry
# -------------------------------------------------
MODEL_NAME = "Hospital_Readmission_Model"
MODEL_VERSION = 1
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

try:
    model = mlflow.xgboost.load_model(MODEL_URI)
    booster = model.get_booster()
    model_features = booster.feature_names
    print("✅ Model loaded successfully!")
    print("🧩 Expected features:", model_features)
except Exception as e:
    print("❌ Error loading model:", e)
    raise e


# -------------------------------------------------
# 🧩 Define Input Schema
# -------------------------------------------------
class PatientData(BaseModel):
    """Schema for patient data input."""
    encounter_id: float
    patient_nbr: float
    time_in_hospital: int
    num_lab_procedures: int
    num_medications: int
    number_outpatient: int
    number_inpatient: int
    number_diagnoses: int
    medical_specialty: float
    discharge_disposition_id: float

    class Config:
        json_schema_extra = {
            "example": {
                "encounter_id": 11001.0,
                "patient_nbr": 21001.0,
                "time_in_hospital": 5,
                "num_lab_procedures": 45,
                "num_medications": 12,
                "number_outpatient": 1,
                "number_inpatient": 1,
                "number_diagnoses": 6,
                "medical_specialty": 1.0,
                "discharge_disposition_id": 1.0
            }
        }


# -------------------------------------------------
# 🌐 API Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    """Health check endpoint to verify API is running."""
    return {"message": "✅ Hospital Readmission Prediction API is live!"}


@app.post("/predict")
def predict(data: PatientData):
    """
    Predict hospital readmission for a patient.

    Args:
        data: PatientData model containing patient features

    Returns:
        Dictionary with prediction class and probabilities

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert Pydantic model to dictionary then DataFrame
        df = pd.DataFrame([data.model_dump()])

        # --- Handle feature order automatically ---
        expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
        df = df.reindex(columns=expected_cols, fill_value=0)

        # --- Get prediction ---
        prediction = int(model.predict(df)[0])

        # --- Handle probability if supported ---
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0].tolist()
        else:
            proba = [float("nan")] * 3  # Placeholder if not available

        return {"prediction": prediction, "probabilities": proba}

    except Exception as e:
        print("❌ Prediction error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
