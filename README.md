# 🏥 Hospital Readmission Prediction — End-to-End MLOps Project  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow)

---

## 🚀 Overview  
This project predicts **hospital patient readmissions** using a complete **end-to-end MLOps pipeline**.  
It combines **data preprocessing**, **model training**, **experiment tracking**, **CI/CD**, and **web deployment** — all built with modern tools like **MLflow**, **FastAPI**, and **Streamlit**.

The goal is to demonstrate how **healthcare analytics** can be transformed into a **production-grade ML solution** using best MLOps practices.

---

## 🧠 Key Highlights  

- ✅ End-to-End MLOps Lifecycle Implementation  
- 📊 ML Experiment Tracking and Model Registry with **MLflow**  
- ⚙️ REST API Deployment using **FastAPI**  
- 💻 Interactive Web Application using **Streamlit**  
- 🧮 Automated Feature Engineering & Model Evaluation  
- ☁️ Ready for Cloud / CI/CD Integration (**GitHub Actions**, **Docker**, **Render**, or **Streamlit Cloud**)  

---

## 🩺 Dataset  
The dataset used in this project represents **hospital discharge records** — including patient demographics, admission types, diagnosis codes, length of stay, and readmission status.  
> *Note:* The data is anonymized and preprocessed for educational and research purposes.  

---

## ⚙️ Tech Stack  

| **Category** | **Tools / Libraries** |
|---------------|-----------------------|
| Programming Language | Python 3.10 |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Modeling | XGBoost, Imbalanced-learn |
| Visualization | Matplotlib, Seaborn |
| MLOps Tools | MLflow, DVC (optional) |
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Version Control & CI/CD | Git, GitHub, YAML pipelines |

---

## 🏗️ Project Architecture  

```bash
Hospital_Readmission_MLOps/
│
├── data/
│   ├── raw/                   # Raw input data
│   ├── processed/             # Processed and cleaned data
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning and transformation
│   ├── feature_selection.py   # Feature importance & selection
│   ├── mlflow_tracking.py     # MLflow experiment tracking
│   ├── mlflow_manual_log.py   # Manual logging with metrics
│   ├── mlflow_model_registry.py # Register best model
│   ├── fastapi_deploy.py      # REST API for real-time inference
│   ├── app_streamlit.py       # Streamlit frontend
│
├── models/                    # Trained and registered MLflow models
├── mlruns/                    # MLflow tracking metadata
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore


## 🔬 Model Workflow  

```text
1. Data Preprocessing – Cleaning missing values, handling categorical encoding, and normalizing numeric columns.  
2. Feature Engineering – Extracting key hospital visit statistics, reducing dimensionality.  
3. Model Training – Training the XGBoost classifier on reduced-feature dataset.  
4. Experiment Tracking – Using MLflow to log metrics (Accuracy, F1, ROC-AUC).  
5. Model Registry – Registering the best performing model version in MLflow.  
6. FastAPI Deployment – Serving real-time predictions through REST endpoints.  
7. Streamlit Frontend – Providing an interactive web UI for hospital staff.

## 📈 Prediction Classes  

```text
0 → Patient will NOT be readmitted  
1 → Patient will be readmitted within 30 days  
2 → Patient will be readmitted after 30 days

