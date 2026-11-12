import os
import sqlite3
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('VERCEL'):
    DB_PATH = '/tmp/smart_lender.db'
else:
    DB_PATH = os.path.join(DB_DIR, 'smart_lender.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT UNIQUE NOT NULL,
            gender TEXT NOT NULL,
            married TEXT NOT NULL,
            dependents TEXT NOT NULL,
            education TEXT NOT NULL,
            self_employed TEXT NOT NULL,
            applicant_income REAL NOT NULL,
            coapplicant_income REAL NOT NULL,
            loan_amount REAL NOT NULL,
            loan_amount_term REAL NOT NULL,
            credit_history REAL NOT NULL,
            property_area TEXT NOT NULL,
            prediction_status TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            model_used TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_application_id ON predictions(application_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON predictions(prediction_status)')
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM predictions')
    count = cursor.fetchone()[0]
    if count == 0:
        seed_data = [
            ('SL-2026-1001', 'Male', 'Yes', '1', 'Graduate', 'No', 5400.0, 2100.0, 160.0, 360.0, 1.0, 'Semiurban', 'Approved', 88.5, 'Random Forest', '2026-11-10 10:15:00'),
            ('SL-2026-1002', 'Female', 'No', '0', 'Graduate', 'Yes', 3200.0, 0.0, 110.0, 360.0, 0.0, 'Urban', 'Rejected', 79.2, 'Random Forest', '2026-11-10 11:30:00'),
            ('SL-2026-1003', 'Male', 'Yes', '2', 'Graduate', 'No', 7800.0, 3500.0, 240.0, 360.0, 1.0, 'Urban', 'Approved', 92.1, 'Random Forest', '2026-11-11 09:20:00'),
            ('SL-2026-1004', 'Male', 'No', '0', 'Not Graduate', 'No', 2500.0, 1200.0, 95.0, 180.0, 1.0, 'Rural', 'Approved', 68.4, 'Random Forest', '2026-11-11 14:45:00'),
            ('SL-2026-1005', 'Female', 'Yes', '1', 'Graduate', 'No', 4100.0, 0.0, 180.0, 360.0, 0.0, 'Rural', 'Rejected', 84.0, 'Random Forest', '2026-11-12 08:10:00')
        ]
        cursor.executemany('''
            INSERT INTO predictions (
                application_id, gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                credit_history, property_area, prediction_status, confidence_score,
                model_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', seed_data)
        conn.commit()
    conn.close()

def save_prediction(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            application_id, gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_amount_term,
            credit_history, property_area, prediction_status, confidence_score,
            model_used, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ''', (
        data['application_id'],
        data['gender'],
        data['married'],
        str(data['dependents']),
        data['education'],
        data['self_employed'],
        float(data['applicant_income']),
        float(data['coapplicant_income']),
        float(data['loan_amount']),
        float(data['loan_amount_term']),
        float(data['credit_history']),
        data['property_area'],
        data['prediction_status'],
        float(data['confidence_score']),
        data['model_used']
    ))
    conn.commit()
    conn.close()
    return data['application_id']

def get_prediction_by_id(application_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE application_id = ?', (application_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_predictions(search=None, status=None, property_area=None, sort_by='newest'):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = 'SELECT * FROM predictions WHERE 1=1'
    params = []

    if search:
        query += ' AND (application_id LIKE ? OR education LIKE ? OR property_area LIKE ?)'
        term = f'%{search}%'
        params.extend([term, term, term])

    if status and status != 'All':
        query += ' AND prediction_status = ?'
        params.append(status)

    if property_area and property_area != 'All':
        query += ' AND property_area = ?'
        params.append(property_area)

    if sort_by == 'oldest':
        query += ' ORDER BY created_at ASC'
    elif sort_by == 'income_high':
        query += ' ORDER BY applicant_income DESC'
    elif sort_by == 'loan_high':
        query += ' ORDER BY loan_amount DESC'
    else:
        query += ' ORDER BY created_at DESC'

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction_status = "Approved"')
    approved = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction_status = "Rejected"')
    rejected = cursor.fetchone()[0]

    approval_rate = round((approved / total * 100), 1) if total > 0 else 0.0

    cursor.execute('SELECT property_area, COUNT(*) FROM predictions GROUP BY property_area')
    area_dist = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT 6')
    recent = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        'total_predictions': total,
        'approved_count': approved,
        'rejected_count': rejected,
        'approval_rate': approval_rate,
        'area_distribution': area_dist,
        'recent_predictions': recent
    }

if __name__ == '__main__':
    init_db()
    print('Database initialized successfully!')
