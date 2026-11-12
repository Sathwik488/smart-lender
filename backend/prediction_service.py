import os
import json
import joblib
import uuid
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global cache for loaded models and metadata
_CACHE = {
    'preprocessor': None,
    'best_model': None,
    'models': {},
    'metadata': None
}

def load_artifacts():
    global _CACHE
    if _CACHE['preprocessor'] is None:
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        best_model_path = os.path.join(MODELS_DIR, 'best_loan_model.joblib')
        metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')

        if os.path.exists(preprocessor_path):
            _CACHE['preprocessor'] = joblib.load(preprocessor_path)

        if os.path.exists(best_model_path):
            _CACHE['best_model'] = joblib.load(best_model_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                _CACHE['metadata'] = json.load(f)

        # Load individual models if available
        model_files = {
            'Decision Tree': 'decision_tree.joblib',
            'Random Forest': 'random_forest.joblib',
            'KNN': 'knn.joblib',
            'Gradient Boosting': 'gradient_boosting.joblib'
        }
        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                _CACHE['models'][name] = joblib.load(path)

    return _CACHE

def get_model_metadata():
    cache = load_artifacts()
    return cache.get('metadata', {})

def validate_input(data):
    errors = []
    cleaned = {}

    # Categorical validations
    gender = str(data.get('gender', '')).strip().capitalize()
    if gender not in ['Male', 'Female']:
        errors.append("Gender must be 'Male' or 'Female'.")
    cleaned['gender'] = gender

    married = str(data.get('married', '')).strip().capitalize()
    if married not in ['Yes', 'No']:
        errors.append("Married status must be 'Yes' or 'No'.")
    cleaned['married'] = married

    dependents = str(data.get('dependents', '')).strip()
    if dependents not in ['0', '1', '2', '3+']:
        errors.append("Dependents must be '0', '1', '2', or '3+'.")
    # For model training, '3+' was mapped to '3'
    cleaned['dependents'] = '3' if dependents == '3+' else dependents
    cleaned['dependents_raw'] = dependents

    education = str(data.get('education', '')).strip()
    if education not in ['Graduate', 'Not Graduate']:
        errors.append("Education must be 'Graduate' or 'Not Graduate'.")
    cleaned['education'] = education

    self_employed = str(data.get('self_employed', '')).strip().capitalize()
    if self_employed not in ['Yes', 'No']:
        errors.append("Self-employed must be 'Yes' or 'No'.")
    cleaned['self_employed'] = self_employed

    property_area = str(data.get('property_area', '')).strip().capitalize()
    if property_area not in ['Urban', 'Semiurban', 'Rural']:
        errors.append("Property Area must be 'Urban', 'Semiurban', or 'Rural'.")
    cleaned['property_area'] = property_area

    # Numerical validations
    try:
        app_income = float(data.get('applicant_income', 0))
        if app_income < 0:
            errors.append("Applicant Income cannot be negative.")
        cleaned['applicant_income'] = app_income
    except (ValueError, TypeError):
        errors.append("Applicant Income must be a valid number.")

    try:
        coapp_income = float(data.get('coapplicant_income', 0))
        if coapp_income < 0:
            errors.append("Co-applicant Income cannot be negative.")
        cleaned['coapplicant_income'] = coapp_income
    except (ValueError, TypeError):
        errors.append("Co-applicant Income must be a valid number.")

    try:
        loan_amount = float(data.get('loan_amount', 0))
        if loan_amount <= 0:
            errors.append("Loan Amount must be greater than 0.")
        cleaned['loan_amount'] = loan_amount
    except (ValueError, TypeError):
        errors.append("Loan Amount must be a valid number.")

    try:
        loan_term = float(data.get('loan_amount_term', 360))
        if loan_term <= 0 or loan_term > 600:
            errors.append("Loan Term must be between 12 and 600 months.")
        cleaned['loan_amount_term'] = loan_term
    except (ValueError, TypeError):
        errors.append("Loan Amount Term must be a valid number.")

    try:
        credit_history = float(data.get('credit_history', 1.0))
        if credit_history not in [0.0, 1.0, 0, 1]:
            errors.append("Credit History must be 1.0 (meets guidelines) or 0.0 (does not meet guidelines).")
        cleaned['credit_history'] = float(credit_history)
    except (ValueError, TypeError):
        errors.append("Credit History must be 0.0 or 1.0.")

    if errors:
        return False, errors, None
    return True, None, cleaned

def predict_loan(input_dict, requested_model_name=None):
    is_valid, errors, data = validate_input(input_dict)
    if not is_valid:
        raise ValueError("; ".join(errors))

    cache = load_artifacts()
    preprocessor = cache['preprocessor']

    if preprocessor is None:
        raise RuntimeError("Machine learning preprocessor is not loaded.")

    # Determine which model to use
    model_name = requested_model_name if requested_model_name in cache['models'] else cache.get('metadata', {}).get('best_model', 'Random Forest')
    model = cache['models'].get(model_name, cache['best_model'])

    if model is None:
        raise RuntimeError(f"Requested model '{model_name}' could not be loaded.")

    # Construct input dataframe matching exact schema expected by preprocessor
    row_df = pd.DataFrame([{
        'Gender': data['gender'],
        'Married': data['married'],
        'Dependents': data['dependents'],
        'Education': data['education'],
        'Self_Employed': data['self_employed'],
        'Property_Area': data['property_area'],
        'ApplicantIncome': data['applicant_income'],
        'CoapplicantIncome': data['coapplicant_income'],
        'LoanAmount': data['loan_amount'],
        'Loan_Amount_Term': data['loan_amount_term'],
        'Credit_History': data['credit_history']
    }])

    # Transform features using pre-fitted preprocessor (no data leakage)
    X_proc = preprocessor.transform(row_df)

    # Perform inference
    prediction = int(model.predict(X_proc)[0])
    
    # Calculate confidence / probability
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_proc)[0]
        # Probability of the predicted class
        confidence = float(probabilities[prediction]) * 100.0
        approval_prob = float(probabilities[1]) * 100.0
    else:
        confidence = 85.0
        approval_prob = 100.0 if prediction == 1 else 0.0

    status = "Approved" if prediction == 1 else "Rejected"

    # Compute financial explainability insights
    total_income = data['applicant_income'] + data['coapplicant_income']
    # Loan amount is in thousands in the dataset
    loan_principal = data['loan_amount'] * 1000.0
    monthly_term = data['loan_amount_term']
    estimated_monthly_emi = round(loan_principal / monthly_term, 2) if monthly_term > 0 else 0.0
    emi_ratio = round((estimated_monthly_emi / (total_income + 1e-5)) * 100, 1)

    # Risk level classification
    if status == "Approved":
        risk_level = "Low Risk" if confidence >= 75.0 else "Moderate Risk"
    else:
        risk_level = "High Risk" if confidence >= 75.0 else "Borderline Risk"

    app_id = f"SL-2026-{uuid.uuid4().hex[:6].upper()}"

    result = {
        'application_id': app_id,
        'prediction_status': status,
        'confidence_score': round(confidence, 1),
        'approval_probability': round(approval_prob, 1),
        'model_used': model_name,
        'risk_level': risk_level,
        'financials': {
            'total_income': total_income,
            'loan_amount_k': data['loan_amount'],
            'loan_principal': loan_principal,
            'loan_term_months': int(monthly_term),
            'estimated_monthly_emi': estimated_monthly_emi,
            'emi_to_income_ratio': emi_ratio,
            'credit_history_status': "Favorable" if data['credit_history'] == 1.0 else "Adverse / Missing"
        },
        'raw_inputs': data
    }

    return result
