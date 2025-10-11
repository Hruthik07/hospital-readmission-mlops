import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------------
# ✅ Must be the first Streamlit command
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🏥 Hospital Readmission Predictor",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------------------------
# ✅ Load Model
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/final_xgboost_model.joblib")
    return model

model = load_model()

# -------------------------------------------------------------------
# ✅ App Title and Description
# -------------------------------------------------------------------
st.title("🏥 Hospital Readmission Prediction (Multiclass Model)")
st.markdown(
    """
    This app predicts the **likelihood of a hospital patient being readmitted**  
    and classifies the readmission as:
    - 🟢 **0:** No Readmission  
    - 🟠 **1:** Readmitted (>30 days)  
    - 🔴 **2:** Readmitted (<30 days)
    """
)

st.write("---")

# -------------------------------------------------------------------
# ✅ Input Fields
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    encounter_id = st.number_input("Encounter ID", min_value=10000, max_value=99999, value=11001)
    patient_nbr = st.number_input("Patient Number", min_value=10000, max_value=99999, value=21001)
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 45)
    num_medications = st.slider("Number of Medications", 1, 50, 12)

with col2:
    number_outpatient = st.slider("Number of Outpatient Visits", 0, 10, 1)
    number_inpatient = st.slider("Number of Inpatient Visits", 0, 10, 1)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 10, 6)
    medical_specialty = st.selectbox("Medical Specialty (encoded)", [1, 2, 3, 4, 5])
    discharge_disposition_id = st.selectbox("Discharge Disposition ID", [1, 2, 3, 4, 5])

# -------------------------------------------------------------------
# ✅ Prepare Input Data
# -------------------------------------------------------------------
input_data = pd.DataFrame(
    {
        "patient_nbr": [patient_nbr],
        "number_inpatient": [number_inpatient],
        "number_diagnoses": [number_diagnoses],
        "time_in_hospital": [time_in_hospital],
        "encounter_id": [encounter_id],
        "discharge_disposition_id": [discharge_disposition_id],
        "num_medications": [num_medications],
        "num_lab_procedures": [num_lab_procedures],
        "medical_specialty": [medical_specialty],
        "number_outpatient": [number_outpatient],
    }
)

st.write("---")

# -------------------------------------------------------------------
# ✅ Class Mapping
# -------------------------------------------------------------------
CLASS_MAP = {
    0: "No readmission",
    1: "Readmitted (>30 days)",
    2: "Readmitted (<30 days)"
}

# -------------------------------------------------------------------
# ✅ Prediction
# -------------------------------------------------------------------
if st.button("🔍 Predict Readmission"):
    try:
        # Predict probabilities for 3 classes
        probs = model.predict_proba(input_data)[0]
        pred_class = int(np.argmax(probs))
        pred_label = CLASS_MAP[pred_class]
        pred_prob = float(probs[pred_class])

        # Display result
        if pred_class == 2:
            st.error(f"🔴 {pred_label} — High Risk (Probability: {pred_prob:.2f})")
        elif pred_class == 1:
            st.warning(f"🟠 {pred_label} — Moderate Risk (Probability: {pred_prob:.2f})")
        else:
            st.success(f"🟢 {pred_label} — Patient is safe (Probability: {pred_prob:.2f})")

        # Show probability table
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": [CLASS_MAP[0], CLASS_MAP[1], CLASS_MAP[2]],
            "Probability": [probs[0], probs[1], probs[2]]
        })
        st.dataframe(prob_df.style.format({"Probability": "{:.2f}"}), use_container_width=True)

        # Show input summary
        st.subheader("📋 Input Summary")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------------------------------------------
# ✅ Footer
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Developed by G.D. Hruthik | MLOps End-to-End Project | Streamlit + MLflow + FastAPI")
