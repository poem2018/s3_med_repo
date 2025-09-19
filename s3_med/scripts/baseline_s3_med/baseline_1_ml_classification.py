#!/usr/bin/env python3
"""
Baseline 1: Traditional ML Classification (Random Forest & Logistic Regression)
Directly train on patient data matrices to predict ICU mortality
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
import argparse
import os
from pathlib import Path

def load_data(data_dir):
    """Load patient data from JSON files"""
    # Load matrices and patient info
    matrices_file = os.path.join(data_dir, 'chartevents_matrices_test.json')
    patient_info_file = os.path.join(data_dir, 'patient_info_test.json')
    
    with open(matrices_file, 'r') as f:
        matrices = json.load(f)
    
    with open(patient_info_file, 'r') as f:
        patient_info = json.load(f)
    
    return matrices, patient_info

def extract_features(matrix, max_hours=48):
    """Extract statistical features from each patient's matrix"""
    features = []
    
    # Process each indicator (17 indicators total)
    for indicator_data in matrix:
        # Limit to first 48 hours
        series = indicator_data[:min(len(indicator_data), max_hours)]
        
        # Convert to numpy array, handling None values
        series_clean = [x for x in series if x is not None and not np.isnan(x)]
        
        if len(series_clean) > 0:
            # Extract statistical features
            features.extend([
                np.mean(series_clean),      # Mean
                np.std(series_clean),       # Standard deviation
                np.min(series_clean),       # Minimum
                np.max(series_clean),       # Maximum
                np.median(series_clean),    # Median
                len(series_clean) / max_hours  # Data availability ratio
            ])
        else:
            # No valid data for this indicator
            features.extend([0, 0, 0, 0, 0, 0])
    
    return features

def prepare_dataset(matrices, patient_info):
    """Prepare feature matrix and labels for ML models"""
    X = []
    y = []
    patient_ids = []
    
    for i, (matrix, info) in enumerate(zip(matrices, patient_info)):
        # Extract features from matrix
        features = extract_features(matrix)
        
        # Add patient demographics
        age = info.get('age', 0)
        gender = 1 if info.get('gender') == 'M' else 0
        
        # Add past medical history features
        num_past_icd = len(info.get('past_icd_codes', []))
        num_past_ccs = len(info.get('past_ccs_codes', []))
        
        # Combine all features
        all_features = features + [age, gender, num_past_icd, num_past_ccs]
        
        X.append(all_features)
        y.append(info.get('mortality', 0))
        patient_ids.append(info.get('patient_id'))
    
    return np.array(X), np.array(y), patient_ids

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'auprc': average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    return metrics, y_pred

def main():
    parser = argparse.ArgumentParser(description="Baseline 1: ML Classification for ICU Mortality")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_1", 
                        help="Directory to save results")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    matrices, patient_info = load_data(args.data_dir)
    
    print(f"Loaded {len(matrices)} patients")
    
    # Prepare dataset
    print("Extracting features...")
    X, y, patient_ids = prepare_dataset(matrices, patient_info)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.sum(y)} deaths out of {len(y)} patients ({np.mean(y)*100:.1f}%)")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=args.random_seed,
        class_weight='balanced'
    )
    rf_metrics, rf_pred = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "RandomForest")
    results.append(rf_metrics)
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.3f}, F1: {rf_metrics['f1']:.3f}, AUC: {rf_metrics['auc']:.3f}, AUPRC: {rf_metrics['auprc']:.3f}")
    
    # 2. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=args.random_seed,
        class_weight='balanced'
    )
    lr_metrics, lr_pred = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "LogisticRegression")
    results.append(lr_metrics)
    print(f"Logistic Regression - Accuracy: {lr_metrics['accuracy']:.3f}, F1: {lr_metrics['f1']:.3f}, AUC: {lr_metrics['auc']:.3f}, AUPRC: {lr_metrics['auprc']:.3f}")
    
    # Feature importance (from Random Forest)
    feature_names = []
    indicators = ['Cap_refill', 'DBP', 'FiO2', 'GCS_eye', 'GCS_motor', 'GCS_total', 
                  'GCS_verbal', 'Glucose', 'HR', 'Height', 'MAP', 'SpO2', 'pH', 
                  'RR', 'SBP', 'Temp', 'Weight']
    
    for ind in indicators:
        feature_names.extend([f"{ind}_mean", f"{ind}_std", f"{ind}_min", 
                             f"{ind}_max", f"{ind}_median", f"{ind}_avail"])
    feature_names.extend(['age', 'gender', 'num_past_icd', 'num_past_ccs'])
    
    # Get top 20 important features
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    print("\nTop 20 Important Features (Random Forest):")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save results
    results_dict = {
        'models': results,
        'feature_importance': {
            'features': [feature_names[idx] for idx in indices],
            'scores': [float(importances[idx]) for idx in indices]
        },
        'test_patients': ids_test,
        'predictions': {
            'RandomForest': rf_pred.tolist(),
            'LogisticRegression': lr_pred.tolist()
        },
        'true_labels': y_test.tolist()
    }
    
    output_file = os.path.join(args.output_dir, 'baseline_1_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE 1: ML CLASSIFICATION RESULTS")
    print("="*50)
    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Accuracy:  {result['accuracy']:.3f}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1 Score:  {result['f1']:.3f}")
        print(f"  AUROC:     {result['auc']:.3f}")
        print(f"  AUPRC:     {result['auprc']:.3f}")

if __name__ == "__main__":
    main()