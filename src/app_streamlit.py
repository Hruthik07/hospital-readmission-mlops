import streamlit as st
import requests
import pandas as pd

# -----------------------------
# 🌐 PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🏥 Hospital Readmission Prediction App",
    layout="centered",
    page_icon="💊"
)

st.title("🏥 Hospital Readmission Prediction System")
st.markdown(
    """
    This app predicts whether a patient will be **readmitted within 30 days**,  
    **after 30 days**, or **not readmitted at all**, using hospital data.
    """
)

# -----------------------------
# 🧾 INPUT FORM
# -----------------------------
with st.form("prediction_form"):
    st.subheader("🧍‍♂️ Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        encounter_id = st.number_input("Encounter ID", min_value=10000, max_value=99999, value=11001)
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 45)
        number_outpatient = st.slider("Number of Outpatient Visits", 0, 10, 1)
        number_inpatient = st.slider("Number of Inpatient Visits", 0, 10, 1)
    with col2:
        patient_nbr = st.number_input("Patient Number", min_value=20000, max_value=99999, value=21001)
        num_medications = st.slider("Number of Medications", 1, 50, 12)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 10, 6)
        medical_specialty = st.selectbox("Medical Specialty (encoded)", [1, 2, 3, 4, 5])
        discharge_disposition_id = st.selectbox("Discharge Disposition ID", [1, 2, 3, 4, 5, 6, 7])

    submitted = st.form_submit_button("🔍 Predict Readmission")

# -----------------------------
# 📦 PREPARE INPUT DATA
# -----------------------------
input_data = {
    "encounter_id": encounter_id,
    "patient_nbr": patient_nbr,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "medical_specialty": medical_specialty,
    "discharge_disposition_id": discharge_disposition_id
}

# -----------------------------
# 🚀 PREDICT
# -----------------------------
if submitted:
    with st.spinner("⏳ Predicting patient outcome..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                probabilities = result.get("probabilities", None)

                # 🧠 Interpret prediction
                if prediction == 0:
                    st.success("🟢 **Patient is not likely to be readmitted.**")
                elif prediction == 1:
                    st.warning("🟠 **Patient is likely to be readmitted within 30 days.**")
                elif prediction == 2:
                    st.error("🔴 **Patient is likely to be readmitted after 30 days.**")
                else:
                    st.info("⚪ Unexpected prediction result.")

                # 📊 Show probability chart if available
                if probabilities:
                    st.markdown("### 📈 Prediction Confidence")
                    prob_df = pd.DataFrame({
                        "Class": ["No Readmission (0)", "Within 30 Days (1)", "After 30 Days (2)"],
                        "Probability": probabilities
                    })

                    st.bar_chart(prob_df.set_index("Class"))
            else:
                st.error(f"Server error: {response.status_code}")

        except Exception as e:
            st.error(f"❌ Error: {e}")