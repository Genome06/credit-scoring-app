# 💳 Predictive Analytics for Home Credit Default Risk (Kaggle Competition)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)
[![ML Model](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning solution designed to predict the probability of a client's loan default. This project bridges the gap between complex data science modeling and real-world deployment using a modular **Object-Oriented Programming (OOP)** architecture.

## 🚀 Project Overview
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. This project utilizes historical loan application data to predict whether an applicant will have difficulties repaying their loan.

### Key Features:
* **Single Prediction Mode:** Interactive UI to input individual applicant data and receive a real-time risk score.
* **Batch Prediction Mode (CSV):** High-volume automated processing allowing users to upload CSV files (e.g., `application_test.csv`) and download full results.
* **Explainable AI (XAI):** Integrated **SHAP Values** to interpret model decisions, ensuring transparency in credit scoring.
* **Smart Risk Rating:** Automated categorization into **Low**, **Medium**, and **High Risk** with actionable business recommendations and system messages.

## 📉 Business Insights & Strategic Decision Making

This project goes beyond technical accuracy by aligning model outputs with core banking KPIs and risk management strategies. The goal is to optimize the balance between **Loan Growth** and **Credit Loss Mitigation**.

### 🔍 1. Key Risk Drivers (Feature Interpretability)
Using SHAP (SHapley Additive exPlanations), the model identifies the most influential factors driving credit default:

* **Financial Health (Payment Rate):** The model heavily penalizes high payment-to-income ratios. High-risk candidates typically have an annuity-to-credit ratio that suggests an unsustainable debt burden.
* **External Credibility (EXT_SOURCES):** Normalized scores from external providers act as a proxy for financial reputation. A low average across these sources is the strongest indicator of potential delinquency.
* **Employment Stability (Days Employed):** The model detects a non-linear relationship where longer tenure in current employment significantly reduces the probability of default, reflecting the importance of income stability.

### 🛡️ 2. Risk-Based Action Plan (The Strategy)
To optimize operational efficiency, the application categorizes applicants into a **Tri-Tier Risk System**:

| Risk Category | Probability Range | Business Strategy | Operational Impact |
| :--- | :--- | :--- | :--- |
| **🟢 Low Risk** | < 30% | **Auto-Approval** | Zero human intervention; maximizes customer experience and reduces operational overhead. |
| **🟡 Medium Risk** | 30% - 60% | **Conditional Offer / Manual Review** | Request additional collateral or verify income via physical documents. Risk-based pricing (higher interest) may be applied. |
| **🔴 High Risk** | > 60% | **Immediate Rejection** | Prevents Non-Performing Loans (NPL) and protects the bank's capital reserves. |

### 📈 3. Strategic Value for Home Credit
* **Expansion to the Unbanked:** By using non-traditional features (like employment duration and goods price ratios), the model allows Home Credit to safely lend to individuals without a traditional credit history.
* **Profitability Optimization:** By accurately identifying Low-Risk individuals, the bank can offer faster approvals and lower interest rates, gaining a competitive edge in the market.
* **Regulatory Compliance (XAI):** Using Explainable AI (SHAP) ensures that the bank can provide a "Reason for Denial" to applicants, fulfilling transparency requirements set by central bank regulations.


## 🛠️ Tech Stack & Architecture

### **Core Technologies**
* **Modeling:** LightGBM (Gradient Boosting Machine) with Hyperparameter Tuning.
* **XAI:** SHAP (SHapley Additive exPlanations).
* **Backend:** FastAPI (High-performance Python web framework).
* **Frontend:** Streamlit (Interactive Data Dashboard).
* **Containerization:** Docker (Multi-process orchestration).

### **Modular OOP Design**
The project follows **Clean Architecture** principles to ensure maintainability and scalability:
* `DataTransformer`: A dedicated class for feature engineering (26 features), robust median imputation, and vectorized log transformations.
* `CreditPredictor`: Manages model inference, risk thresholding logic, and business insight mapping.
* `Schemas`: Utilizes **Pydantic** for rigorous data validation and automatic Swagger documentation.

---

## 📂 Directory Structure

```text
credit-scoring-app/
├── app/
│   ├── api/                # FastAPI Endpoints & Pydantic Schemas
│   ├── core/               # OOP Logic (Preprocessor & Predictor)
│   └── frontend/           # Streamlit UI Components
├── artifacts/              # Serialized Models (.joblib) & Scalers
├── Dockerfile              # Containerization Script
├── requirements.txt        # Project Dependencies
├── run.sh                  # Orchestration Script (Internal API & External UI)
└── README.md
```

## 📊 Model Interpretability (XAI)

This model is built on transparency. Using SHAP Values, the system identifies key risk drivers for each prediction. Global analysis shows that EXT_SOURCE ratings, PAYMENT_RATE, and DAYS_EMPLOYED are the most significant factors in determining repayment probability.

![Shap Value](image.png)

## ⚙️ Installation & Local Setup

1. Clone the Repository
```
git clone [https://github.com/yourusername/home-credit-risk-app.git](https://github.com/yourusername/home-credit-risk-app.git)
cd home-credit-risk-app
```
2. Manual Setup (Local)
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Docker Setup (Recommended)
```
docker build -t credit-scoring-app .
docker run -p 7860:7860 credit-scoring-app
```
Access the UI at http://localhost:7860.

## 🌐 Deployment
This application is designed to be deployed on Hugging Face Spaces. It uses a custom run.sh entrypoint to orchestrate both the FastAPI backend (internal) and the Streamlit frontend (public) within a single Docker container.

**Disclaimer: This project is a personal portfolio developed for educational purposes. It utilizes the public dataset provided by Home Credit on Kaggle. This application is not affiliated with, endorsed by, or an official product of Home Credit Group.**


---
title: Home Dredit Risk Analysis
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---