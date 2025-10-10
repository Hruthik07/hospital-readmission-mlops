from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import traceback

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


# -------------------------------------------------
# 🌐 API Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "✅ Hospital Readmission Prediction API is live!"}


@app.post("/predict")
def predict(data: dict):
    import numpy as np
    try:
        df = pd.DataFrame([data])

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
        import traceback
        print("❌ Prediction error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
