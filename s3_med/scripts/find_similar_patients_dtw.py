#!/usr/bin/env python3
"""
Find similar patient cases based on DTW (Dynamic Time Warping) similarity
of chartevents matrices and merge with temporal patient data
"""

import json
import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define the 17 indicators in order
INDICATORS = [
    'Capillary refill rate',
    'Diastolic blood pressure',
    'Fraction inspired oxygen',
    'Glascow coma scale eye opening',
    'Glascow coma scale motor response',
    'Glascow coma scale total',
    'Glascow coma scale verbal response',
    'Glucose',
    'Heart Rate',
    'Height',
    'Mean blood pressure',
    'Oxygen saturation',
    'pH',
    'Respiratory rate',
    'Systolic blood pressure',
    'Temperature',
    'Weight'
]

# Indicator weights for similarity calculation
INDICATOR_WEIGHTS = {
    0: 0.1,   # Capillary refill rate (常缺失)
    1: 1.0,   # Diastolic blood pressure
    2: 0.5,   # FiO2 (辅助)
    3: 0.7,   # GCS eye
    4: 1.5,   # GCS motor ⭐ (最重要)
    5: 0.3,   # GCS total (冗余)
    6: 0.7,   # GCS verbal
    7: 1.0,   # Glucose
    8: 1.2,   # Heart Rate
    9: 0.05,   # Height (静态数据，权重改为0.1)
    10: 1.5,  # MAP ⭐ (关键)
    11: 1.5,  # SpO2 ⭐ (关键)
    12: 0.8,  # pH
    13: 1.2,  # Respiratory rate
    14: 1.0,  # Systolic BP
    15: 0.3,  # Temperature
    16: 0.05,  # Weight (静态数据，权重改为0.1)
}

def load_data():
    """Load all necessary data files"""
    data_dir = '/scratch/bcew/ruikez2/intern/s3_med/data'
    
    # Load chartevents matrices
    with open(os.path.join(data_dir, 'chartevents_matrices.json'), 'r') as f:
        matrices = json.load(f)
    
    # Load patient info  
    with open(os.path.join(data_dir, 'patient_info.json'), 'r') as f:
        patient_info = json.load(f)
    
    # Load temporal patient data
    with open(os.path.join(data_dir, 'patient_data_with_temporal.json'), 'r') as f:
        temporal_data = json.load(f)
    
    return matrices, patient_info, temporal_data

def preprocess_series(series, max_hours=48):
    """Preprocess a time series: handle missing values and limit to 48 hours"""
    # Limit to first 48 hours
    series = series[:min(len(series), max_hours)]
    
    # Handle None values by forward fill then backward fill
    processed = []
    last_valid = None
    
    for val in series:
        if val is not None:
            processed.append(float(val))
            last_valid = float(val)
        else:
            if last_valid is not None:
                processed.append(last_valid)
            else:
                processed.append(0.0)  # If no valid value yet, use 0
    
    # If series is too short, pad with last value
    if len(processed) < 3:  # DTW needs at least some data points
        if processed:
            processed = processed + [processed[-1]] * (3 - len(processed))
        else:
            processed = [0.0, 0.0, 0.0]
    
    return np.array(processed)

def normalize_series(series):
    """Normalize a series to [0, 1] range"""
    if len(series) == 0:
        return series
    
    min_val = np.min(series)
    max_val = np.max(series)
    
    if max_val - min_val > 0:
        return (series - min_val) / (max_val - min_val)
    else:
        return series * 0.0  # All same value -> return zeros

def calculate_dtw_distance(series1, series2):
    """Calculate DTW distance between two time series"""
    # Preprocess series
    s1 = preprocess_series(series1)
    s2 = preprocess_series(series2)
    
    # Skip if either series is all zeros
    if np.all(s1 == 0) or np.all(s2 == 0):
        return float('inf')
    
    # Normalize for fair comparison
    s1_norm = normalize_series(s1)
    s2_norm = normalize_series(s2)
    
    try:
        # Calculate DTW distance
        distance = dtw.distance(s1_norm, s2_norm)
        return distance
    except:
        return float('inf')

def calculate_multivariate_dtw(matrix1, matrix2):
    """
    Calculate multivariate DTW distance between two patient matrices
    Using weighted combination of all 17 indicator DTW distances
    """
    distances = []
    weights = []
    
    # Use all 17 indicators with their specific weights
    for idx in range(min(len(matrix1), len(matrix2), 17)):
        weight = INDICATOR_WEIGHTS.get(idx, 0.1)
        
        # Skip indicators with very low weight (near 0)
        if weight < 0.05:
            continue
            
        dist = calculate_dtw_distance(matrix1[idx], matrix2[idx])
        
        if dist != float('inf'):
            distances.append(dist)
            weights.append(weight)
    
    if distances:
        # Weighted average of distances
        weighted_dist = np.average(distances, weights=weights)
        return weighted_dist
    else:
        return float('inf')

def find_most_similar_patients(target_idx, all_matrices, patient_info, n_similar=1, exclude_self=True):
    """Find the n most similar patients based on DTW similarity"""
    target_matrix = all_matrices[target_idx]
    target_mortality = patient_info[target_idx]['mortality']
    
    similarities = []
    
    for i, matrix in enumerate(all_matrices):
        if exclude_self and i == target_idx:
            continue
        
        # Calculate DTW distance
        distance = calculate_multivariate_dtw(target_matrix, matrix)
        
        if distance != float('inf'):
            # Convert distance to similarity score (inverse)
            similarity = 1.0 / (1.0 + distance)
            
            # Bonus for same mortality outcome (helps with clinical relevance)
            if patient_info[i]['mortality'] == target_mortality:
                similarity *= 1.2  # 20% bonus
            
            similarities.append((i, similarity, distance))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n similar patients
    return similarities[:n_similar]

def create_similar_patient_summary(patient_info, temporal_data):
    """Create a detailed summary string for a similar patient case with ALL indicators"""
    demographics = patient_info
    
    # Extract statistics for ALL 17 indicators
    vital_signs_summary = []
    
    # 0. Capillary refill rate
    crr_values = extract_values_from_temporal(temporal_data.get('Capillary refill rate', ''))
    if crr_values:
        vital_signs_summary.append(f"Capillary refill: {np.mean(crr_values):.1f}")
    
    # 1. Diastolic blood pressure
    dbp_values = extract_values_from_temporal(temporal_data.get('Diastolic blood pressure', ''))
    if dbp_values:
        vital_signs_summary.append(f"DBP {np.min(dbp_values):.0f}-{np.max(dbp_values):.0f} mmHg")
    
    # 2. Fraction inspired oxygen
    fio2_values = extract_values_from_temporal(temporal_data.get('Fraction inspired oxygen', ''))
    if fio2_values:
        vital_signs_summary.append(f"FiO2 {np.min(fio2_values):.1f}-{np.max(fio2_values):.1f}")
    
    # 3-6. Glasgow Coma Scale components
    gcs_eye_values = extract_values_from_temporal(temporal_data.get('Glascow coma scale eye opening', ''))
    if gcs_eye_values:
        vital_signs_summary.append(f"GCS Eye: {max(set(gcs_eye_values), key=gcs_eye_values.count):.0f}")
    
    gcs_motor_values = extract_values_from_temporal(temporal_data.get('Glascow coma scale motor response', ''))
    if gcs_motor_values:
        mode_gcs = max(set(gcs_motor_values), key=gcs_motor_values.count)
        if mode_gcs == 1:
            vital_signs_summary.append("GCS Motor: 1 (no response)")
        else:
            vital_signs_summary.append(f"GCS Motor: {mode_gcs:.0f}")
    
    gcs_total_values = extract_values_from_temporal(temporal_data.get('Glascow coma scale total', ''))
    if gcs_total_values:
        vital_signs_summary.append(f"GCS Total: {np.mean(gcs_total_values):.0f}")
    
    gcs_verbal_values = extract_values_from_temporal(temporal_data.get('Glascow coma scale verbal response', ''))
    if gcs_verbal_values:
        vital_signs_summary.append(f"GCS Verbal: {max(set(gcs_verbal_values), key=gcs_verbal_values.count):.0f}")
    
    # 7. Glucose
    glucose_values = extract_values_from_temporal(temporal_data.get('Glucose', ''))
    if glucose_values:
        glucose_range = f"Glucose {np.min(glucose_values):.0f}-{np.max(glucose_values):.0f} mg/dL"
        if np.max(glucose_values) > 200:
            glucose_range += " (high)"
        vital_signs_summary.append(glucose_range)
    
    # 8. Heart Rate
    hr_values = extract_values_from_temporal(temporal_data.get('Heart Rate', ''))
    if hr_values:
        hr_text = f"HR {np.mean(hr_values):.0f} bpm"
        if np.max(hr_values) > 120:
            hr_text += f" (max {np.max(hr_values):.0f})"
        vital_signs_summary.append(hr_text)
    
    # 9. Height
    height_values = extract_values_from_temporal(temporal_data.get('Height', ''))
    if height_values:
        vital_signs_summary.append(f"Height {np.mean(height_values):.0f} cm")
    
    # 10. Mean blood pressure
    map_values = extract_values_from_temporal(temporal_data.get('Mean blood pressure', ''))
    if map_values:
        map_min = np.min(map_values)
        if map_min < 65:
            vital_signs_summary.append(f"MAP <65 mmHg (low)")
        else:
            vital_signs_summary.append(f"MAP {map_min:.0f}-{np.max(map_values):.0f} mmHg")
    
    # 11. Oxygen saturation
    spo2_values = extract_values_from_temporal(temporal_data.get('Oxygen saturation', ''))
    if spo2_values:
        vital_signs_summary.append(f"SpO2 {np.min(spo2_values):.0f}-{np.max(spo2_values):.0f}%")
    
    # 12. pH
    ph_values = extract_values_from_temporal(temporal_data.get('pH', ''))
    if ph_values:
        ph_min = np.min(ph_values)
        ph_max = np.max(ph_values)
        ph_text = f"pH {ph_min:.2f}-{ph_max:.2f}"
        if ph_min < 7.35:
            ph_text += " (acidosis)"
        elif ph_max > 7.45:
            ph_text += " (alkalosis)"
        vital_signs_summary.append(ph_text)
    
    # 13. Respiratory rate
    rr_values = extract_values_from_temporal(temporal_data.get('Respiratory rate', ''))
    if rr_values:
        vital_signs_summary.append(f"RR {np.mean(rr_values):.0f} breaths/min")
    
    # 14. Systolic blood pressure
    sbp_values = extract_values_from_temporal(temporal_data.get('Systolic blood pressure', ''))
    if sbp_values:
        vital_signs_summary.append(f"SBP {np.min(sbp_values):.0f}-{np.max(sbp_values):.0f} mmHg")
    
    # 15. Temperature
    temp_values = extract_values_from_temporal(temporal_data.get('Temperature', ''))
    if temp_values:
        temp_text = f"Temp {np.mean(temp_values):.1f}°C"
        if np.max(temp_values) > 38:
            temp_text += " (fever)"
        vital_signs_summary.append(temp_text)
    
    # 16. Weight
    weight_values = extract_values_from_temporal(temporal_data.get('Weight', ''))
    if weight_values:
        vital_signs_summary.append(f"Weight {np.mean(weight_values):.1f} kg")
    
    # Build summary text
    summary_lines = [
        "Similar Patient Case:",
        f"- Age: {demographics['age']} years",
        f"- Gender: {demographics['gender']}",
        f"- Patient ID: SIMILAR_{demographics['patient_id'][-6:]}",
    ]
    
    # Add past medical history - directly from patient_info
    if demographics.get('past_ccs_codes') and len(demographics['past_ccs_codes']) > 0:
        past_ccs = demographics['past_ccs_codes'][:10]  # Limit to first 10
        summary_lines.append(f"- Past CCS codes: {', '.join(past_ccs)}")
    else:
        summary_lines.append("- Past CCS codes: None")
    
    summary_lines.append("")
    summary_lines.append("Clinical Measurements (0-48h summary):")
    
    # Add vital signs
    for vs in vital_signs_summary:
        summary_lines.append(f"- {vs}")
    
    # Add outcome
    mortality_text = "died" if demographics['mortality'] == 1 else "survived"
    summary_lines.append("")
    summary_lines.append(f"This similar patient {mortality_text} (In-hospital mortality = {demographics['mortality']}).")
    
    return "\n".join(summary_lines)

def extract_values_from_temporal(temporal_string):
    """Extract numerical values from temporal data string"""
    if not temporal_string or temporal_string == "No data available":
        return []
    
    values = []
    parts = temporal_string.split(', ')
    for part in parts:
        try:
            value = float(part.split(' at hour')[0])
            values.append(value)
        except:
            continue
    
    return values

def main():
    print("Loading data...")
    matrices, patient_info, temporal_patients = load_data()
    
    print(f"Loaded {len(matrices)} matrices")
    print(f"Loaded {len(patient_info)} patient records")
    print(f"Loaded {len(temporal_patients)} temporal patient records")
    
    # Ensure we have matching data
    min_length = min(len(matrices), len(patient_info), len(temporal_patients))
    print(f"Processing {min_length} patients with complete data")
    
    # Find similar patients for each patient
    print("\nFinding similar patients using DTW...")
    print("(This may take a while for large datasets)")
    patients_with_similar = []
    
    for i in range(min_length):
        patient = temporal_patients[i].copy()
        
        # Find most similar patient
        similar_patients = find_most_similar_patients(i, matrices, patient_info, n_similar=1)
        
        if similar_patients:
            similar_idx, similarity_score, dtw_distance = similar_patients[0]
            
            # Create similar patient summary
            similar_summary = create_similar_patient_summary(
                patient_info[similar_idx],
                temporal_patients[similar_idx]['temporal_data']
            )
            
            # Add similar patient case to the patient data
            patient['similar_patient_case'] = similar_summary
            patient['similar_patient_id'] = patient_info[similar_idx]['patient_id']
            patient['similarity_score'] = float(similarity_score)
            patient['dtw_distance'] = float(dtw_distance)
        else:
            patient['similar_patient_case'] = "No similar patient found"
            patient['similar_patient_id'] = None
            patient['similarity_score'] = 0.0
            patient['dtw_distance'] = float('inf')
        
        patients_with_similar.append(patient)
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{min_length} patients")
    
    # Save the result
    output_path = '/scratch/bcew/ruikez2/intern/s3_med/data/patient_data_with_similar_dtw.json'
    print(f"\nSaving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(patients_with_similar, f, indent=2)
    
    print(f"Successfully saved {len(patients_with_similar)} patients with similar cases")
    
    # Print statistics
    mortality_stats = {}
    similarity_by_outcome = {'same': [], 'different': []}
    
    for patient in patients_with_similar:
        mortality = patient['mortality']
        if mortality not in mortality_stats:
            mortality_stats[mortality] = 0
        mortality_stats[mortality] += 1
        
        # Check similarity scores
        if patient.get('similar_patient_id'):
            # Find similar patient's mortality
            for p in patient_info:
                if p['patient_id'] == patient['similar_patient_id']:
                    if p['mortality'] == mortality:
                        similarity_by_outcome['same'].append(patient['similarity_score'])
                    else:
                        similarity_by_outcome['different'].append(patient['similarity_score'])
                    break
    
    print("\nMortality distribution:")
    print(f"  Survived (mortality=0): {mortality_stats.get(0, 0)}")
    print(f"  Died (mortality=1): {mortality_stats.get(1, 0)}")
    
    print("\nSimilarity score statistics:")
    if similarity_by_outcome['same']:
        print(f"  Same outcome: avg similarity = {np.mean(similarity_by_outcome['same']):.3f}")
    if similarity_by_outcome['different']:
        print(f"  Different outcome: avg similarity = {np.mean(similarity_by_outcome['different']):.3f}")
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS:")
    print("="*80)
    
    # Show one survived and one died patient
    for mortality_val in [0, 1]:
        for patient in patients_with_similar[:50]:  # Check first 50 for samples
            if patient['mortality'] == mortality_val and patient.get('similar_patient_case'):
                print(f"\nPatient ID: {patient['patient_id']}")
                print(f"Mortality: {patient['mortality']}")
                print(f"DTW Distance: {patient.get('dtw_distance', 'N/A'):.3f}")
                print(f"Similarity Score: {patient.get('similarity_score', 0):.3f}")
                print(f"\n{patient['similar_patient_case']}")
                print("-"*40)
                break
    
    print("="*80)

if __name__ == "__main__":
    main()