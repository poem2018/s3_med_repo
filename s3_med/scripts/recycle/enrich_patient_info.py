#!/usr/bin/env python3
"""
Enrich patient info with demographics and diagnosis information
"""

import json
import gzip
from datetime import datetime
from collections import defaultdict
import os
import sys
import csv

def load_patient_info(patient_info_file):
    """Load existing patient info"""
    with open(patient_info_file, 'r') as f:
        return json.load(f)

def load_patients_data(patients_file):
    """Load patient demographics"""
    patients = {}
    
    with gzip.open(patients_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        gender_idx = header.index('gender')
        anchor_age_idx = header.index('anchor_age')
        anchor_year_idx = header.index('anchor_year')
        dod_idx = header.index('dod') if 'dod' in header else None
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(subject_idx, gender_idx, anchor_age_idx, anchor_year_idx):
                subject_id = parts[subject_idx]
                patients[subject_id] = {
                    'gender': parts[gender_idx],
                    'anchor_age': int(parts[anchor_age_idx]) if parts[anchor_age_idx] else None,
                    'anchor_year': int(parts[anchor_year_idx]) if parts[anchor_year_idx] else None,
                    'dod': parts[dod_idx] if dod_idx and len(parts) > dod_idx and parts[dod_idx] else None
                }
    
    return patients

def load_admissions_data(admissions_file):
    """Load admission information"""
    admissions = defaultdict(dict)
    
    with gzip.open(admissions_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        hadm_idx = header.index('hadm_id')
        admittime_idx = header.index('admittime')
        admission_type_idx = header.index('admission_type')
        insurance_idx = header.index('insurance')
        language_idx = header.index('language')
        marital_status_idx = header.index('marital_status')
        race_idx = header.index('race')
        hospital_expire_idx = header.index('hospital_expire_flag')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(subject_idx, hadm_idx, admittime_idx):
                subject_id = parts[subject_idx]
                hadm_id = parts[hadm_idx]
                admissions[subject_id][hadm_id] = {
                    'admittime': parts[admittime_idx],
                    'admission_type': parts[admission_type_idx] if len(parts) > admission_type_idx else '',
                    'insurance': parts[insurance_idx] if len(parts) > insurance_idx else '',
                    'language': parts[language_idx] if len(parts) > language_idx else '',
                    'marital_status': parts[marital_status_idx] if len(parts) > marital_status_idx else '',
                    'race': parts[race_idx] if len(parts) > race_idx else '',
                    'hospital_expire_flag': int(parts[hospital_expire_idx]) if len(parts) > hospital_expire_idx and parts[hospital_expire_idx].isdigit() else 0
                }
    
    return admissions

def load_icustays_data(icustays_file):
    """Load ICU stay to hospital admission mapping"""
    icustay_to_hadm = {}
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        hadm_idx = header.index('hadm_id')
        stay_idx = header.index('stay_id')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(subject_idx, hadm_idx, stay_idx):
                subject_id = parts[subject_idx]
                stay_id = parts[stay_idx]
                hadm_id = parts[hadm_idx]
                patient_key = f"{subject_id}_{stay_id}"
                icustay_to_hadm[patient_key] = (subject_id, hadm_id)
    
    return icustay_to_hadm

def load_diagnoses_data(diagnoses_file):
    """Load diagnosis ICD codes"""
    diagnoses = defaultdict(lambda: defaultdict(list))
    
    with gzip.open(diagnoses_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        hadm_idx = header.index('hadm_id')
        seq_num_idx = header.index('seq_num')
        icd_code_idx = header.index('icd_code')
        icd_version_idx = header.index('icd_version')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(subject_idx, hadm_idx, seq_num_idx, icd_code_idx, icd_version_idx):
                subject_id = parts[subject_idx]
                hadm_id = parts[hadm_idx]
                seq_num = int(parts[seq_num_idx])
                icd_code = parts[icd_code_idx]
                icd_version = int(parts[icd_version_idx])
                
                diagnoses[subject_id][hadm_id].append({
                    'seq_num': seq_num,
                    'icd_code': icd_code,
                    'icd_version': icd_version
                })
    
    return diagnoses

def load_icd_descriptions(d_icd_file):
    """Load ICD code descriptions"""
    icd_descriptions = {}
    
    with gzip.open(d_icd_file, 'rt') as f:
        header = f.readline().strip().split(',')
        code_idx = header.index('icd_code')
        version_idx = header.index('icd_version')
        title_idx = header.index('long_title')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(code_idx, version_idx, title_idx):
                icd_code = parts[code_idx]
                icd_version = int(parts[version_idx])
                title = parts[title_idx]
                icd_descriptions[(icd_code, icd_version)] = title
    
    return icd_descriptions

def calculate_age_at_icu(anchor_age, anchor_year, icu_start_time):
    """Calculate patient age at ICU admission"""
    if not anchor_age or not anchor_year or not icu_start_time:
        return None
    
    try:
        icu_year = datetime.strptime(icu_start_time, '%Y-%m-%d %H:%M:%S').year
        age_at_icu = anchor_age + (icu_year - anchor_year)
        return age_at_icu
    except:
        return None

def load_icd_to_ccs_mappings():
    """Load both ICD9 and ICD10 to CCS mappings"""
    icd9_to_ccs = {}
    icd10_to_ccs = {}
    
    # Load ICD9 mappings
    icd9_file = '/scratch/bcew/ruikez2/intern/KARE/kg_construct/resources/ICD9CM_to_CCSCM.csv'
    print("Loading ICD9 to CCS mappings...")
    with open(icd9_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            icd9_to_ccs[row['ICD9CM']] = row['CCSCM']
    
    # Load ICD10 mappings
    icd10_file = '/scratch/bcew/ruikez2/intern/KARE/ehr_prepare/resources/ICD10CM_to_CCSCM.csv'
    print("Loading ICD10 to CCS mappings...")
    with open(icd10_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            icd10_to_ccs[row['ICD10CM']] = row['CCSCM']
    
    print(f"Loaded {len(icd9_to_ccs)} ICD9 mappings and {len(icd10_to_ccs)} ICD10 mappings")
    return icd9_to_ccs, icd10_to_ccs

def load_ccs_descriptions():
    """Load CCS code descriptions"""
    ccs_descriptions = {}
    
    ccs_file = '/scratch/bcew/ruikez2/intern/KARE/ehr_prepare/resources/CCSCM.csv'
    print("Loading CCS descriptions...")
    with open(ccs_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ccs_descriptions[row['code']] = row['name']
    
    print(f"Loaded {len(ccs_descriptions)} CCS descriptions")
    return ccs_descriptions

def convert_icd_to_ccs(icd_code, icd_version, icd9_mappings, icd10_mappings):
    """Convert an ICD code to CCS code based on version"""
    if icd_version == 9:
        # Try original code first
        ccs = icd9_mappings.get(icd_code)
        
        if not ccs:
            # Try adding decimal point for ICD9 codes
            # ICD9 codes typically have format XXX.XX
            if len(icd_code) > 3:
                formatted_code = icd_code[:3] + '.' + icd_code[3:]
                ccs = icd9_mappings.get(formatted_code)
            
            # Try without leading zeros
            if not ccs:
                clean_code = icd_code.lstrip('0')
                ccs = icd9_mappings.get(clean_code)
                
            # Try adding decimal to cleaned code
            if not ccs and len(clean_code) > 3:
                formatted_clean = clean_code[:3] + '.' + clean_code[3:]
                ccs = icd9_mappings.get(formatted_clean)
        
        return ccs
    elif icd_version == 10:
        # Try original code first
        ccs = icd10_mappings.get(icd_code)
        
        if not ccs:
            # Try removing periods
            clean_code = icd_code.replace('.', '')
            ccs = icd10_mappings.get(clean_code)
            
        if not ccs:
            # Try adding period for ICD10 codes (typically after 3rd character)
            if len(icd_code) > 3 and '.' not in icd_code:
                formatted_code = icd_code[:3] + '.' + icd_code[3:]
                ccs = icd10_mappings.get(formatted_code)
        
        return ccs
    else:
        return None

def get_historical_visits(subject_id, current_hadm_id, current_admit_time, admissions_data, diagnoses_data, icd_descriptions, icd9_mappings, icd10_mappings, ccs_descriptions):
    """Get all historical visits for a patient before current admission"""
    historical_visits = []
    
    if subject_id in admissions_data:
        # Get all admissions for this patient
        all_admissions = []
        for hadm_id, adm_info in admissions_data[subject_id].items():
            if hadm_id != current_hadm_id:  # Exclude current admission
                try:
                    admit_time = datetime.strptime(adm_info['admittime'], '%Y-%m-%d %H:%M:%S')
                    current_time = datetime.strptime(current_admit_time, '%Y-%m-%d %H:%M:%S')
                    
                    # Only include admissions before current one
                    if admit_time < current_time:
                        all_admissions.append({
                            'hadm_id': hadm_id,
                            'admittime': adm_info['admittime'],
                            'admission_type': adm_info['admission_type'],
                            'admit_datetime': admit_time
                        })
                except:
                    continue
        
        # Sort by admission time (most recent first)
        all_admissions.sort(key=lambda x: x['admit_datetime'], reverse=True)
        
        # Get diagnoses for each historical admission
        for adm in all_admissions[:10]:  # Limit to 10 most recent admissions
            visit_info = {
                'hadm_id': adm['hadm_id'],
                'admittime': adm['admittime'],
                'admission_type': adm['admission_type'],
                'diagnoses': []
            }
            
            # Get diagnoses for this admission
            if subject_id in diagnoses_data and adm['hadm_id'] in diagnoses_data[subject_id]:
                diags = diagnoses_data[subject_id][adm['hadm_id']]
                diags.sort(key=lambda x: x['seq_num'])
                
                # Process first 5 diagnoses
                for diag in diags[:5]:
                    # Convert to CCS
                    ccs_code = convert_icd_to_ccs(
                        diag['icd_code'],
                        diag['icd_version'],
                        icd9_mappings,
                        icd10_mappings
                    )
                    
                    diag_info = {
                        'icd_code': diag['icd_code'],
                        'icd_version': diag['icd_version'],
                        'seq_num': diag['seq_num'],
                        'description': icd_descriptions.get(
                            (diag['icd_code'], diag['icd_version']), 
                            'Unknown'
                        ),
                        'ccs_code': ccs_code if ccs_code else None,
                        'ccs_desc': ccs_descriptions.get(ccs_code, 'No CCS mapping found') if ccs_code else 'No CCS mapping found'
                    }
                    visit_info['diagnoses'].append(diag_info)
            
            historical_visits.append(visit_info)
    
    return historical_visits

def enrich_patient_info(patient_info_file, output_file, data_dir):
    """Add demographics and diagnosis info to patient records"""
    
    # Load all necessary data
    print("Loading existing patient info...")
    patient_info = load_patient_info(patient_info_file)
    
    print("Loading patient demographics...")
    patients_data = load_patients_data(os.path.join(data_dir, 'hosp/patients.csv.gz'))
    
    print("Loading admissions data...")
    admissions_data = load_admissions_data(os.path.join(data_dir, 'hosp/admissions.csv.gz'))
    
    print("Loading ICU stays mapping...")
    icustay_to_hadm = load_icustays_data(os.path.join(data_dir, 'icu/icustays.csv.gz'))
    
    print("Loading diagnoses...")
    diagnoses_data = load_diagnoses_data(os.path.join(data_dir, 'hosp/diagnoses_icd.csv.gz'))
    
    print("Loading ICD descriptions...")
    icd_descriptions = load_icd_descriptions(os.path.join(data_dir, 'hosp/d_icd_diagnoses.csv.gz'))
    
    # Load CCS mappings
    print("\nLoading CCS mappings...")
    icd9_mappings, icd10_mappings = load_icd_to_ccs_mappings()
    ccs_descriptions = load_ccs_descriptions()
    
    # Enrich each patient
    print("\nEnriching patient information...")
    for patient in patient_info:
        patient_id = patient['patient_id']
        
        # Get subject_id and hadm_id
        if patient_id in icustay_to_hadm:
            subject_id, hadm_id = icustay_to_hadm[patient_id]
            
            # Add demographics
            if subject_id in patients_data:
                demo = patients_data[subject_id]
                patient['gender'] = demo['gender']
                patient['age_at_icu'] = calculate_age_at_icu(
                    demo['anchor_age'], 
                    demo['anchor_year'], 
                    patient['start_time']
                )
                patient['mortality'] = 1 if demo['dod'] else 0
            
            # Add admission info
            if subject_id in admissions_data and hadm_id in admissions_data[subject_id]:
                adm = admissions_data[subject_id][hadm_id]
                patient['admission_type'] = adm['admission_type']
                patient['language'] = adm['language']
                patient['marital_status'] = adm['marital_status']
                patient['race'] = adm['race']
                patient['hospital_expire_flag'] = adm['hospital_expire_flag']
            
            # Add diagnoses
            if subject_id in diagnoses_data and hadm_id in diagnoses_data[subject_id]:
                diags = diagnoses_data[subject_id][hadm_id]
                # Sort by sequence number
                diags.sort(key=lambda x: x['seq_num'])
                
                # Add primary diagnosis (seq_num = 1)
                primary_diag = None
                for diag in diags:
                    if diag['seq_num'] == 1:
                        primary_diag = diag
                        break
                
                if primary_diag:
                    patient['primary_diagnosis_code'] = primary_diag['icd_code']
                    patient['primary_diagnosis_version'] = primary_diag['icd_version']
                    
                    # Get description
                    desc_key = (primary_diag['icd_code'], primary_diag['icd_version'])
                    if desc_key in icd_descriptions:
                        patient['primary_diagnosis_desc'] = icd_descriptions[desc_key]
                    else:
                        patient['primary_diagnosis_desc'] = 'Unknown'
                    
                    # Convert to CCS
                    ccs_code = convert_icd_to_ccs(
                        primary_diag['icd_code'],
                        primary_diag['icd_version'],
                        icd9_mappings,
                        icd10_mappings
                    )
                    
                    if ccs_code:
                        patient['primary_ccs_code'] = ccs_code
                        patient['primary_ccs_desc'] = ccs_descriptions.get(ccs_code, 'Unknown CCS category')
                    else:
                        patient['primary_ccs_code'] = None
                        patient['primary_ccs_desc'] = 'No CCS mapping found'
                
                # Add all diagnosis codes with CCS conversion
                all_diagnoses = []
                for diag in diags[:10]:  # Limit to first 10 diagnoses
                    # Convert to CCS
                    ccs_code = convert_icd_to_ccs(
                        diag['icd_code'],
                        diag['icd_version'],
                        icd9_mappings,
                        icd10_mappings
                    )
                    
                    diag_info = {
                        'code': diag['icd_code'],
                        'version': diag['icd_version'],
                        'seq_num': diag['seq_num'],
                        'description': icd_descriptions.get(
                            (diag['icd_code'], diag['icd_version']), 
                            'Unknown'
                        ),
                        'ccs_code': ccs_code if ccs_code else None,
                        'ccs_desc': ccs_descriptions.get(ccs_code, 'No CCS mapping found') if ccs_code else 'No CCS mapping found'
                    }
                    all_diagnoses.append(diag_info)
                
                patient['all_diagnoses'] = all_diagnoses
            
            # Add historical visits
            current_admit_time = adm['admittime'] if subject_id in admissions_data and hadm_id in admissions_data[subject_id] else None
            if current_admit_time:
                patient['historical_visits'] = get_historical_visits(
                    subject_id, 
                    hadm_id, 
                    current_admit_time,
                    admissions_data,
                    diagnoses_data,
                    icd_descriptions,
                    icd9_mappings,
                    icd10_mappings,
                    ccs_descriptions
                )
    
    # Save enriched data
    with open(output_file, 'w') as f:
        json.dump(patient_info, f, indent=2)
    
    print(f"\nEnriched patient info saved to {output_file}")
    
    # Print summary with CCS stats
    print("\nConversion Statistics:")
    total_diagnoses = 0
    successful_conversions = 0
    unique_ccs_codes = set()
    
    for patient in patient_info:
        if 'primary_ccs_code' in patient and patient['primary_ccs_code']:
            successful_conversions += 1
            unique_ccs_codes.add(patient['primary_ccs_code'])
        if 'all_diagnoses' in patient:
            for diag in patient['all_diagnoses']:
                total_diagnoses += 1
                if diag.get('ccs_code'):
                    successful_conversions += 1
                    unique_ccs_codes.add(diag['ccs_code'])
    
    print(f"  Total diagnoses processed: {total_diagnoses}")
    print(f"  Successful CCS conversions: {successful_conversions}")
    if total_diagnoses > 0:
        print(f"  Conversion success rate: {successful_conversions / total_diagnoses * 100:.1f}%")
    print(f"  Unique CCS codes found: {len(unique_ccs_codes)}")
    
    print("\nSample enriched patient info:")
    if patient_info:
        sample = patient_info[0]
        for key in ['patient_id', 'gender', 'age_at_icu', 'race', 'primary_diagnosis_desc', 'primary_ccs_code', 'primary_ccs_desc']:
            if key in sample:
                print(f"  {key}: {sample[key]}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python enrich_patient_info.py <patient_info_dir>")
        print("Example: python enrich_patient_info.py mimic_matrices_10_patients")
        sys.exit(1)
    
    patient_dir = sys.argv[1]
    patient_info_file = os.path.join(patient_dir, 'patient_info.json')
    output_file = os.path.join(patient_dir, 'patient_info_enriched.json')
    data_dir = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1'
    
    if not os.path.exists(patient_info_file):
        print(f"Error: {patient_info_file} not found")
        sys.exit(1)
    
    enrich_patient_info(patient_info_file, output_file, data_dir)

if __name__ == "__main__":
    main()