# Smart Lender – An Intelligent Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.3-teal)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange)
![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.3-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Live Deployment](https://img.shields.io/badge/Live%20Demo-smart--lender--wheat.vercel.app-brightgreen?style=for-the-badge&logo=vercel)
![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)

> **Live Website:** [https://smart-lender-wheat.vercel.app](https://smart-lender-wheat.vercel.app)  
> **GitHub Repository:** [https://github.com/Sathwik488/smart-lender](https://github.com/Sathwik488/smart-lender)  
> **Render 1-Click Deploy:** [Deploy on Render](https://render.com/deploy?repo=https://github.com/Sathwik488/smart-lender)

**Smart Lender** is an end-to-end, full-stack Machine Learning web application developed as a Capstone Academic Final-Year Project. It assists financial institutions, underwriters, and credit applicants by evaluating creditworthiness and predicting loan approval odds in real time using balanced machine learning algorithms.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Machine Learning Workflow](#machine-learning-workflow)
3. [Algorithm Benchmark Results](#algorithm-benchmark-results)
4. [Technology Stack](#technology-stack)
5. [Directory Structure](#directory-structure)
6. [Local Installation & Setup](#local-installation--setup)
7. [Running Automated Tests](#running-automated-tests)
8. [REST API Documentation](#rest-api-documentation)
9. [Deployment](#deployment)
10. [Academic Disclaimer](#academic-disclaimer)

---

## 1. Key Features

- **Modern FinTech Interface**: Engineered with a responsive Navy Blue & Teal palette, clean Inter typography, and intuitive form controls.
- **Dynamic Application Form**: Instant client-side validation and live financial calculators for Combined Household Income and Estimated Monthly EMI.
- **Zero Data Leakage Pipeline**: Scikit-Learn `ColumnTransformer` handles mode imputation for categorical columns, median imputation for numerical features, one-hot encoding, and standard scaling strictly within the training workflow.
- **SMOTE Class Imbalance Handling**: Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training split, preventing model bias toward majority approvals.
- **Multi-Algorithm Support**: Evaluates 4 distinct classification models (Random Forest, Gradient Boosting, Decision Tree, and K-Nearest Neighbors).
- **Interactive Underwriting Dashboard**: Live KPIs, approval distribution donut charts, and model benchmark comparisons powered by Chart.js.
- **Searchable Audit History**: Comprehensive SQLite database persistence with real-time filtering, search, and sorting.
- **Production REST APIs**: Exposes endpoints for headless integration, automated batch scoring, and health monitoring.

---

## 2. Machine Learning Workflow

```
[ Kaggle / AV Dataset (614 records) ]
                  │
                  ▼
   [ Stratified 80/20 Train-Test Split ]
                  │
                  ├──────────────────────────────┐
                  ▼                              ▼
      [ Train Set (491 rows) ]        [ Test Set (123 rows) ]
                  │                              │
                  ▼                              ▼
       [ Preprocessor Fit ]          [ Preprocessor Transform ]
  - Mode Imputer + OneHotEncoder                  │
  - Median Imputer + StandardScaler               │
                  │                              │
                  ▼                              │
      [ SMOTE Class Balancing ]                  │
  - Rejected: 154 -> 337                         │
  - Approved: 337 -> 337                         │
                  │                              │
                  ▼                              │
       [ Train 4 ML Models ]                     │
  - Random Forest                                │
  - Gradient Boosting                            │
  - Decision Tree                                │
  - K-Nearest Neighbors                          │
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                     [ Evaluate on Test Set ]
                   (No Leakage, Real Metrics)
                                 │
                                 ▼
                  [ Select Best: Random Forest ]
               (Accuracy: 79.67%, F1-Score: 85.71%)
```

---

## 3. Algorithm Benchmark Results

The models were evaluated on the stratified unseen test set ($N=123$):

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **79.67%** | **83.33%** | **88.24%** | **85.71%** | **0.8198** | **Best Model (Selected)** |
| Gradient Boosting | 76.42% | 81.11% | 85.88% | 83.43% | 0.7766 | Competitive |
| K-Nearest Neighbors (KNN) | 75.61% | 82.35% | 82.35% | 82.35% | 0.8266 | High Precision |
| Decision Tree Classifier | 74.80% | 79.35% | 85.88% | 82.49% | 0.6814 | Baseline |

---

## 4. Technology Stack

- **Backend**: Python 3, Flask, Gunicorn
- **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE), Pandas, NumPy, Joblib
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript (ES6), Bootstrap 5.3, Chart.js, FontAwesome 6
- **Deployment**: Render, GitHub

---

## 5. Directory Structure

```
smart-lender/
├── app.py                      # Application entry point and Flask routing
├── requirements.txt            # Python dependencies
├── Procfile                    # Gunicorn production start command
├── render.yaml                 # Render cloud configuration
├── test_app.py                 # Automated unit and integration tests
├── README.md                   # Project documentation
├── dataset/
│   └── loan_data.csv           # Authentic 614-record loan prediction dataset
├── machine_learning/
│   └── train_and_evaluate.py   # Training, SMOTE balancing, & evaluation pipeline
├── models/
│   ├── best_loan_model.joblib  # Serialized best model (Random Forest)
│   ├── preprocessor.joblib     # Serialized ColumnTransformer pipeline
│   ├── model_metrics.json      # Empirical benchmark results
│   ├── decision_tree.joblib    # Individual trained model artifact
│   ├── random_forest.joblib    # Individual trained model artifact
│   ├── knn.joblib              # Individual trained model artifact
│   └── gradient_boosting.joblib# Individual trained model artifact
├── database/
│   ├── db.py                   # SQLite schema, queries, and connection handling
│   └── smart_lender.db         # Persistent SQLite database
├── backend/
│   └── prediction_service.py   # Validation, inference, & financial metrics
└── frontend/
    ├── templates/
    │   ├── base.html           # Unified responsive layout
    │   ├── index.html          # Hero, workflow, features, & CTA
    │   ├── about.html          # Academic background & dataset schema
    │   ├── predict.html        # Dynamic loan application form
    │   ├── result.html         # Status, confidence gauge, & breakdown
    │   ├── dashboard.html      # Interactive Chart.js analytics
    │   ├── history.html        # Searchable and filterable audit history
    │   └── contact.html        # Project metadata and feedback form
    └── static/
        ├── css/custom.css      # Custom FinTech theme
        ├── js/main.js          # Dynamic form handling & client validation
        ├── js/charts.js        # Chart.js visualization logic
        └── images/             # Generated plots (confusion matrix, ROC, etc.)
```

---

## 6. Local Installation & Setup

### Prerequisites
- Python 3.10+ installed
- Git

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sathwik488/smart-lender.git
   cd smart-lender
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models (optional, pre-trained artifacts included)**:
   ```bash
   python machine_learning/train_and_evaluate.py
   ```

5. **Initialize Database**:
   ```bash
   python database/db.py
   ```

6. **Run the Application**:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

---

## 7. Running Automated Tests

Run the test suite covering data preprocessing, model inference, SQLite operations, and Flask routes:

```bash
python -m unittest test_app.py
```

Expected output:
```
......
----------------------------------------------------------------------
Ran 6 tests in 0.239s

OK
```

---

## 8. REST API Documentation

### 1. Model Inference
- **Endpoint**: `POST /api/predict`
- **Request Body**:
  ```json
  {
    "gender": "Male",
    "married": "Yes",
    "dependents": "1",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 5849,
    "coapplicant_income": 2000,
    "loan_amount": 150,
    "loan_amount_term": 360,
    "credit_history": 1.0,
    "property_area": "Semiurban",
    "model_choice": "Random Forest"
  }
  ```
- **Response** (`201 Created`):
  ```json
  {
    "success": true,
    "data": {
      "application_id": "SL-2026-A1B2C3",
      "prediction_status": "Approved",
      "confidence_score": 88.5,
      "risk_level": "Low Risk",
      "model_used": "Random Forest"
    }
  }
  ```

### 2. Live Dashboard Statistics
- **Endpoint**: `GET /api/statistics`

### 3. Historical Submissions
- **Endpoint**: `GET /api/history`

### 4. Health Check
- **Endpoint**: `GET /api/health`

---

## 9. Deployment

### Render Deployment
This repository is pre-configured with `render.yaml` and `Procfile`.
1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com).
3. Click **New +** -> **Web Service**.
4. Select the `smart-lender` repository.
5. Environment: `Python 3`.
6. Build Command: `pip install -r requirements.txt && python machine_learning/train_and_evaluate.py && python database/db.py`.
7. Start Command: `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`.
8. Click **Deploy Web Service**.

---

## 10. Academic Disclaimer

This project is developed solely for educational and academic evaluation. The machine learning predictions are derived from historical patterns and should not be considered binding legal financial advice or formal underwriting decisions.
