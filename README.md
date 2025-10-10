# 🏥 Hospital Readmission Prediction — End-to-End MLOps Project

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
