# API Documentation

## Hospital Readmission Prediction API

### Overview
This FastAPI-based REST API provides predictions for hospital readmissions using a trained XGBoost model.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
**GET** `/`

Returns a simple health check message to verify the API is running.

**Response:**
```json
{
  "message": "✅ Hospital Readmission Prediction API is live!"
}
```

#### 2. Predict Readmission
**POST** `/predict`

Makes a prediction for hospital readmission based on patient data.

**Request Body:**
```json
{
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
```

**Request Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| encounter_id | float | Unique encounter identifier |
| patient_nbr | float | Unique patient identifier |
| time_in_hospital | int | Length of hospital stay in days |
| num_lab_procedures | int | Number of laboratory procedures performed |
| num_medications | int | Number of medications prescribed |
| number_outpatient | int | Number of outpatient visits in the past year |
| number_inpatient | int | Number of inpatient visits in the past year |
| number_diagnoses | int | Number of diagnoses |
| medical_specialty | float | Medical specialty (encoded) |
| discharge_disposition_id | float | Discharge disposition identifier |

**Response:**
```json
{
  "prediction": 0,
  "probabilities": [0.85, 0.10, 0.05]
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| prediction | int | Predicted class (0=No readmission, 1=Readmitted >30 days, 2=Readmitted <30 days) |
| probabilities | array | Probability for each class [Class 0, Class 1, Class 2] |

**Error Response:**
```json
{
  "detail": "Prediction failed: <error_message>"
}
```

### Running the API

#### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.fastapi_deploy:app --reload --host 0.0.0.0 --port 8000
```

#### Access Interactive Documentation
Once the API is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example Usage

#### Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Model Information
- **Model Type:** XGBoost Classifier
- **Model Version:** 1
- **Registry Name:** Hospital_Readmission_Model
- **Target Classes:** 
  - 0: No readmission
  - 1: Readmitted after 30 days
  - 2: Readmitted within 30 days

### Notes
- The API expects all features in the correct order
- Missing features will be filled with 0
- Ensure the MLflow model registry is accessible
- Model must be trained and registered before API use
