import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'loan_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    static_img_dir = os.path.join(base_dir, 'frontend', 'static', 'images')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)

    print('[*] Loading dataset from:', dataset_path)
    df = pd.read_csv(dataset_path)

    # 1. Exploratory Data Cleaning
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Clean Dependents: treat '3+' as '3' or string categorical
    df['Dependents'] = df['Dependents'].astype(str).str.strip()
    df.loc[df['Dependents'] == '3+', 'Dependents'] = '3'
    df.loc[df['Dependents'] == 'nan', 'Dependents'] = np.nan

    # Target variable mapping: Y -> 1 (Approved), N -> 0 (Rejected)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df = df.dropna(subset=['Loan_Status'])
    df['Loan_Status'] = df['Loan_Status'].astype(int)

    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    X = df[categorical_features + numerical_features].copy()
    y = df['Loan_Status'].values

    # Train-test split before ANY preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f'[*] Training samples: {len(X_train)}, Testing samples: {len(X_test)}')
    print(f'[*] Class distribution in train before SMOTE: Rejected={sum(y_train == 0)}, Approved={sum(y_train == 1)}')

    # 2. Build Preprocessing Pipelines
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, categorical_features),
            ('num', num_pipeline, numerical_features)
        ],
        remainder='drop'
    )

    # Fit preprocessor strictly on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_encoded_cols = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = cat_encoded_cols + numerical_features

    # 3. Apply SMOTE exclusively on training set to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)

    print(f'[*] Class distribution in train after SMOTE: Rejected={sum(y_train_resampled == 0)}, Approved={sum(y_train_resampled == 1)}')

    # 4. Define Candidate ML Algorithms
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_split=6, min_samples_leaf=2, random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_score = -1.0

    print('============================================================')
    print('MODEL TRAINING & EVALUATION RESULTS')
    print('============================================================')

    for name, model in models.items():
        # Train model on resampled training data
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on raw test data (preprocessed using fitted preprocessor)
        y_pred = model.predict(X_test_proc)
        
        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_proc)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        else:
            roc_auc = None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
            'confusion_matrix': cm
        }

        print(f'[{name}]')
        print(f'  Accuracy  : {acc*100:.2f}%')
        print(f'  Precision : {prec*100:.2f}%')
        print(f'  Recall    : {rec*100:.2f}%')
        print(f'  F1-Score  : {f1*100:.2f}%')
        if roc_auc:
            print(f'  ROC-AUC   : {roc_auc:.4f}')
        print(f'  Confusion Matrix: {cm}')

        # Save individual model
        slug = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(models_dir, f'{slug}.joblib'))

        # Composite score to select best model (favoring balanced F1 and Accuracy)
        composite_score = (f1 * 0.6) + (acc * 0.4)
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name

    print('============================================================')
    print(f'[*] BEST PERFORMING MODEL: {best_model_name} (Composite Score: {best_score:.4f})')
    print('============================================================')

    # Save best model and preprocessor
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, 'best_loan_model.joblib'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))

    # Save feature names and metadata
    metadata = {
        'best_model': best_model_name,
        'algorithms': results,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'encoded_feature_names': all_feature_names,
        'test_set_size': int(len(y_test)),
        'train_set_size': int(len(y_train))
    }

    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    # 5. Generate Visualizations for Frontend and Reports
    sns.set_theme(style='whitegrid')
    
    # Chart 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(models))
    width = 0.18

    colors = ['#0d9488', '#3b82f6', '#8b5cf6', '#10b981']
    for idx, metric in enumerate(metrics_names):
        vals = [results[m][metric] for m in models]
        plt.bar(x + idx*width, vals, width, label=metric.replace('_', ' ').capitalize(), color=colors[idx])

    plt.xlabel('Machine Learning Models', fontweight='bold', fontsize=11)
    plt.ylabel('Score', fontweight='bold', fontsize=11)
    plt.title('Algorithm Performance Comparison', fontweight='bold', fontsize=13)
    plt.xticks(x + width * 1.5, list(models.keys()), fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(static_img_dir, 'model_comparison.png'), dpi=200)
    plt.close()

    # Chart 2: Confusion Matrix for Best Model
    best_cm = np.array(results[best_model_name]['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'])
    plt.title(f'Confusion Matrix: {best_model_name}', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_img_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    # Chart 3: Feature Importances (if supported)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_names = [all_feature_names[i] for i in indices]
        top_scores = importances[indices]

        plt.figure(figsize=(9, 5))
        sns.barplot(x=top_scores, y=top_names, hue=top_names, palette='viridis', legend=False)
        plt.title(f'Top 10 Feature Importances ({best_model_name})', fontweight='bold', fontsize=12)
        plt.xlabel('Relative Importance', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(static_img_dir, 'feature_importance.png'), dpi=200)
        plt.close()

    print(f'[OK] All artifacts saved to {models_dir}')
    print(f'[OK] Visualizations saved to {static_img_dir}')
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
