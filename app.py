import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from database.db import (
    init_db, save_prediction, get_prediction_by_id,
    get_all_predictions, get_dashboard_stats
)
from backend.prediction_service import (
    predict_loan, validate_input, get_model_metadata, load_artifacts
)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'smart-lender-secret-key-2026')

# Initialize DB and preload ML models on app startup
with app.app_context():
    init_db()
    load_artifacts()

@app.route('/')
def index():
    stats = get_dashboard_stats()
    metadata = get_model_metadata()
    return render_template('index.html', stats=stats, metadata=metadata, active_page='home')

@app.route('/about')
def about():
    metadata = get_model_metadata()
    return render_template('about.html', metadata=metadata, active_page='about')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    metadata = get_model_metadata()
    if request.method == 'POST':
        form_data = {
            'gender': request.form.get('gender'),
            'married': request.form.get('married'),
            'dependents': request.form.get('dependents'),
            'education': request.form.get('education'),
            'self_employed': request.form.get('self_employed'),
            'applicant_income': request.form.get('applicant_income'),
            'coapplicant_income': request.form.get('coapplicant_income'),
            'loan_amount': request.form.get('loan_amount'),
            'loan_amount_term': request.form.get('loan_amount_term'),
            'credit_history': request.form.get('credit_history'),
            'property_area': request.form.get('property_area')
        }
        selected_model = request.form.get('model_choice') or None

        is_valid, errors, cleaned = validate_input(form_data)
        if not is_valid:
            for err in errors:
                flash(err, 'danger')
            return render_template('predict.html', form=form_data, metadata=metadata, active_page='predict')

        try:
            result = predict_loan(form_data, requested_model_name=selected_model)
            
            # Save prediction to SQLite database
            db_record = {
                'application_id': result['application_id'],
                'gender': cleaned['gender'],
                'married': cleaned['married'],
                'dependents': cleaned['dependents_raw'],
                'education': cleaned['education'],
                'self_employed': cleaned['self_employed'],
                'applicant_income': cleaned['applicant_income'],
                'coapplicant_income': cleaned['coapplicant_income'],
                'loan_amount': cleaned['loan_amount'],
                'loan_amount_term': cleaned['loan_amount_term'],
                'credit_history': cleaned['credit_history'],
                'property_area': cleaned['property_area'],
                'prediction_status': result['prediction_status'],
                'confidence_score': result['confidence_score'],
                'model_used': result['model_used']
            }
            save_prediction(db_record)

            return redirect(url_for('prediction_result', application_id=result['application_id']))
        except Exception as e:
            flash(f"Error processing prediction: {str(e)}", 'danger')
            return render_template('predict.html', form=form_data, metadata=metadata, active_page='predict')

    return render_template('predict.html', form={}, metadata=metadata, active_page='predict')

@app.route('/result/<application_id>')
def prediction_result(application_id):
    prediction = get_prediction_by_id(application_id)
    if not prediction:
        flash(f"Application ID '{application_id}' not found.", 'warning')
        return redirect(url_for('history'))

    # Compute additional financial indicators for display
    total_income = prediction['applicant_income'] + prediction['coapplicant_income']
    loan_principal = prediction['loan_amount'] * 1000.0
    loan_term = prediction['loan_amount_term']
    monthly_emi = round(loan_principal / loan_term, 2) if loan_term > 0 else 0.0
    emi_ratio = round((monthly_emi / (total_income + 1e-5)) * 100, 1)

    financials = {
        'total_income': total_income,
        'loan_principal': loan_principal,
        'monthly_emi': monthly_emi,
        'emi_ratio': emi_ratio,
        'credit_status': "Meets Prime Guidelines (1.0)" if prediction['credit_history'] == 1.0 else "Sub-prime / Delinquent (0.0)"
    }

    return render_template('result.html', p=prediction, financials=financials, active_page='predict')

@app.route('/dashboard')
def dashboard():
    stats = get_dashboard_stats()
    metadata = get_model_metadata()
    return render_template('dashboard.html', stats=stats, metadata=metadata, active_page='dashboard')

@app.route('/history')
def history():
    search = request.args.get('search', '').strip()
    status = request.args.get('status', 'All')
    property_area = request.args.get('property_area', 'All')
    sort_by = request.args.get('sort_by', 'newest')

    predictions = get_all_predictions(search=search, status=status, property_area=property_area, sort_by=sort_by)
    return render_template(
        'history.html',
        predictions=predictions,
        search=search,
        status=status,
        property_area=property_area,
        sort_by=sort_by,
        active_page='history'
    )

@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

# ----------------- REST API Endpoints ----------------- #

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON request payload'}), 400

    is_valid, errors, cleaned = validate_input(data)
    if not is_valid:
        return jsonify({'success': False, 'errors': errors}), 422

    try:
        model_name = data.get('model_choice')
        result = predict_loan(data, requested_model_name=model_name)

        db_record = {
            'application_id': result['application_id'],
            'gender': cleaned['gender'],
            'married': cleaned['married'],
            'dependents': cleaned['dependents_raw'],
            'education': cleaned['education'],
            'self_employed': cleaned['self_employed'],
            'applicant_income': cleaned['applicant_income'],
            'coapplicant_income': cleaned['coapplicant_income'],
            'loan_amount': cleaned['loan_amount'],
            'loan_amount_term': cleaned['loan_amount_term'],
            'credit_history': cleaned['credit_history'],
            'property_area': cleaned['property_area'],
            'prediction_status': result['prediction_status'],
            'confidence_score': result['confidence_score'],
            'model_used': result['model_used']
        }
        save_prediction(db_record)

        return jsonify({'success': True, 'data': result}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    stats = get_dashboard_stats()
    return jsonify({'success': True, 'data': stats})

@app.route('/api/history', methods=['GET'])
def api_history():
    search = request.args.get('search')
    status = request.args.get('status')
    property_area = request.args.get('property_area')
    sort_by = request.args.get('sort_by', 'newest')

    predictions = get_all_predictions(search=search, status=status, property_area=property_area, sort_by=sort_by)
    return jsonify({'success': True, 'count': len(predictions), 'data': predictions})

@app.route('/api/models', methods=['GET'])
def api_models():
    metadata = get_model_metadata()
    return jsonify({'success': True, 'data': metadata})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'smart-lender-api', 'version': '1.0.0'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
