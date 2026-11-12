import unittest
import os
import json
from app import app
from database.db import (
    init_db, save_prediction, get_prediction_by_id,
    get_all_predictions, get_dashboard_stats
)
from backend.prediction_service import predict_loan, validate_input

class SmartLenderTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        init_db()

    def test_model_artifacts_exist(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'models', 'best_loan_model.joblib')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'models', 'preprocessor.joblib')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'models', 'model_metrics.json')))

    def test_database_crud(self):
        sample = {
            'application_id': 'TEST-UNIT-001',
            'gender': 'Male',
            'married': 'Yes',
            'dependents': '0',
            'education': 'Graduate',
            'self_employed': 'No',
            'applicant_income': 6000.0,
            'coapplicant_income': 2500.0,
            'loan_amount': 140.0,
            'loan_amount_term': 360.0,
            'credit_history': 1.0,
            'property_area': 'Semiurban',
            'prediction_status': 'Approved',
            'confidence_score': 85.5,
            'model_used': 'Random Forest'
        }
        app_id = save_prediction(sample)
        self.assertEqual(app_id, 'TEST-UNIT-001')

        record = get_prediction_by_id('TEST-UNIT-001')
        self.assertIsNotNone(record)
        self.assertEqual(record['prediction_status'], 'Approved')
        self.assertEqual(record['applicant_income'], 6000.0)

        all_records = get_all_predictions(search='TEST-UNIT-001')
        self.assertGreaterEqual(len(all_records), 1)

        stats = get_dashboard_stats()
        self.assertGreater(stats['total_predictions'], 0)
        self.assertIn('approved_count', stats)
        self.assertIn('rejected_count', stats)

    def test_prediction_inference(self):
        sample = {
            'gender': 'Male',
            'married': 'Yes',
            'dependents': '1',
            'education': 'Graduate',
            'self_employed': 'No',
            'applicant_income': 5849,
            'coapplicant_income': 2000,
            'loan_amount': 150,
            'loan_amount_term': 360,
            'credit_history': 1.0,
            'property_area': 'Semiurban'
        }
        res = predict_loan(sample)
        self.assertIn('prediction_status', res)
        self.assertIn(res['prediction_status'], ['Approved', 'Rejected'])
        self.assertGreaterEqual(res['confidence_score'], 0.0)
        self.assertLessEqual(res['confidence_score'], 100.0)
        self.assertTrue(res['application_id'].startswith('SL-2026-'))

    def test_prediction_validation_errors(self):
        invalid_sample = {
            'gender': 'Unknown',
            'applicant_income': -500,
            'loan_amount': -10
        }
        is_valid, errors, cleaned = validate_input(invalid_sample)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 3)

    def test_web_routes(self):
        routes = ['/', '/about', '/predict', '/dashboard', '/history', '/contact']
        for r in routes:
            response = self.client.get(r)
            self.assertEqual(response.status_code, 200, f"Route {r} failed with status {response.status_code}")

    def test_api_endpoints(self):
        # Health check
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'healthy')

        # Model comparison metrics
        res_models = self.client.get('/api/models')
        self.assertEqual(res_models.status_code, 200)
        self.assertIn('algorithms', res_models.get_json()['data'])

        # API Predict
        payload = {
            'gender': 'Female',
            'married': 'No',
            'dependents': '0',
            'education': 'Graduate',
            'self_employed': 'No',
            'applicant_income': 4500,
            'coapplicant_income': 0,
            'loan_amount': 120,
            'loan_amount_term': 360,
            'credit_history': 1.0,
            'property_area': 'Urban'
        }
        res_pred = self.client.post('/api/predict', json=payload)
        self.assertEqual(res_pred.status_code, 201)
        pred_data = res_pred.get_json()
        self.assertTrue(pred_data['success'])
        self.assertIn('prediction_status', pred_data['data'])

if __name__ == '__main__':
    unittest.main()
