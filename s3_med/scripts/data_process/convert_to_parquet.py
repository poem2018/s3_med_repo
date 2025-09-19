#!/usr/bin/env python3
"""
Convert JSON data to Parquet format for MIMIC ICU mortality prediction
"""

import json
import pandas as pd
import sys

# The three data samples provided
json_data = [
    {
        "text": "<question>\nBased on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.\n</question>\n\n<patient_data>\nPatient Demographics:\n- Age: 79 years\n- Gender: Male\n- Patient ID: 10037861_34531557\n\nPast Medical History:\n- Acute and unspecified renal failure\n- Bacterial infection; unspecified site\n- Cardiac dysrhythmias\n- Chronic kidney disease\n- Coagulation and hemorrhagic disorders\n- Congestive heart failure; nonhypertensive\n- Coronary atherosclerosis and other heart disease\n- Diabetes mellitus without complication\n- Disorders of lipid metabolism\n- Fluid and electrolyte disorders\n- Hypertension with complications and secondary hypertension\n- Other aftercare\n- Other circulatory disease\n- Pulmonary heart disease\n- Screening and history of mental health and substance abuse codes\n- Urinary tract infections\n\nClinical Measurements (First 48 hours):\n\nDiastolic blood pressure:\n  54.50 at hour 5, 54.00 at hour 6, 80.00 at hour 7, 61.00 at hour 8, 66.00 at hour 9, 58.00 at hour 10, 50.00 at hour 11, 47.00 at hour 12, 51.00 at hour 13, 53.00 at hour 14, 50.00 at hour 15, 51.00 at hour 16, 54.00 at hour 17, 55.00 at hour 18, 52.00 at hour 19, 54.00 at hour 20, 56.00 at hour 21, 54.00 at hour 22, 64.00 at hour 23, 55.00 at hour 24, 57.00 at hour 25, 55.00 at hour 26, 59.00 at hour 27, 56.00 at hour 28, 48.00 at hour 29, 47.00 at hour 30, 53.00 at hour 31, 50.00 at hour 32, 61.00 at hour 33, 50.00 at hour 34, 52.00 at hour 35, 50.00 at hour 36, 45.00 at hour 37, 46.00 at hour 38, 55.00 at hour 39, 53.00 at hour 40, 60.00 at hour 41, 58.00 at hour 42, 52.00 at hour 43, 52.00 at hour 44, 43.00 at hour 45, 45.00 at hour 46, 53.00 at hour 47, 54.00 at hour 48\n\nFraction inspired oxygen:\n  0.50 at hour 5, 0.40 at hour 7, 0.40 at hour 12, 0.40 at hour 13, 0.40 at hour 16, 0.40 at hour 19, 0.40 at hour 25, 0.40 at hour 29, 0.40 at hour 32, 0.30 at hour 36, 0.30 at hour 37, 0.30 at hour 40, 0.30 at hour 43, 0.30 at hour 48\n\nGlascow coma scale eye opening:\n  2.00 at hour 20, 2.00 at hour 22, 3.00 at hour 24, 3.00 at hour 26, 3.00 at hour 28, 3.00 at hour 30, 3.00 at hour 32, 3.00 at hour 34, 3.00 at hour 36, 3.00 at hour 38\n\nGlascow coma scale motor response:\n  1.00 at hour 5, 1.00 at hour 6, 1.00 at hour 7, 1.00 at hour 8, 1.00 at hour 9, 1.00 at hour 10, 1.00 at hour 11, 1.00 at hour 12, 1.00 at hour 13, 1.00 at hour 14, 1.00 at hour 15, 1.00 at hour 16, 1.00 at hour 17, 1.00 at hour 18, 6.00 at hour 20, 6.00 at hour 22, 6.00 at hour 24, 6.00 at hour 26, 6.00 at hour 28, 6.00 at hour 30, 6.00 at hour 32, 6.00 at hour 34, 6.00 at hour 36, 6.00 at hour 38, 6.00 at hour 41, 6.00 at hour 42, 6.00 at hour 44, 6.00 at hour 46, 6.00 at hour 48\n\nGlascow coma scale verbal response:\n  1.00 at hour 5, 1.00 at hour 6, 1.00 at hour 7, 1.00 at hour 8, 1.00 at hour 9, 1.00 at hour 10, 1.00 at hour 11, 1.00 at hour 12, 1.00 at hour 13, 1.00 at hour 14, 1.00 at hour 15, 1.00 at hour 16, 1.00 at hour 17, 1.00 at hour 18, 1.00 at hour 20, 1.00 at hour 22, 1.00 at hour 24, 1.00 at hour 26, 1.00 at hour 28, 1.00 at hour 30, 1.00 at hour 32, 1.00 at hour 34, 1.00 at hour 36, 1.00 at hour 38, 1.00 at hour 41, 1.00 at hour 42, 1.00 at hour 44, 1.00 at hour 46, 1.00 at hour 48\n\nGlucose:\n  166.00 at hour 5, 168.00 at hour 10, 165.00 at hour 18, 168.00 at hour 24, 204.00 at hour 30, 264.00 at hour 33, 217.00 at hour 42, 261.00 at hour 48\n\nHeart Rate:\n  67.00 at hour 5, 62.00 at hour 6, 81.00 at hour 7, 79.00 at hour 8, 79.00 at hour 9, 93.00 at hour 10, 83.00 at hour 11, 94.00 at hour 12, 93.00 at hour 13, 93.00 at hour 14, 91.00 at hour 15, 88.00 at hour 16, 85.00 at hour 17, 87.00 at hour 18, 92.00 at hour 19, 76.00 at hour 20, 88.00 at hour 21, 77.00 at hour 22, 84.00 at hour 23, 73.00 at hour 24, 85.00 at hour 25, 74.00 at hour 26, 80.00 at hour 27, 82.00 at hour 28, 100.00 at hour 29, 94.00 at hour 30, 84.00 at hour 31, 90.00 at hour 32, 84.00 at hour 33, 80.00 at hour 34, 82.00 at hour 35, 87.00 at hour 36, 81.00 at hour 37, 81.00 at hour 38, 84.00 at hour 39, 87.00 at hour 40, 89.00 at hour 41, 100.00 at hour 42, 81.00 at hour 43, 97.00 at hour 44, 102.00 at hour 45, 95.00 at hour 46, 93.00 at hour 47, 93.00 at hour 48\n\nMean blood pressure:\n  72.50 at hour 5, 68.50 at hour 6, 105.00 at hour 7, 81.00 at hour 8, 84.00 at hour 9, 77.00 at hour 10, 67.00 at hour 11, 61.00 at hour 12, 69.00 at hour 13, 72.00 at hour 14, 68.00 at hour 15, 71.00 at hour 16, 72.00 at hour 17, 76.00 at hour 18, 72.00 at hour 19, 70.00 at hour 20, 73.00 at hour 21, 71.00 at hour 22, 90.00 at hour 23, 73.00 at hour 24, 75.00 at hour 25, 73.00 at hour 26, 77.00 at hour 27, 75.00 at hour 28, 65.00 at hour 29, 62.00 at hour 30, 70.00 at hour 31, 68.00 at hour 32, 79.00 at hour 33, 65.00 at hour 34, 70.00 at hour 35, 68.00 at hour 36, 64.00 at hour 37, 62.00 at hour 38, 75.00 at hour 39, 70.00 at hour 40, 83.00 at hour 41, 81.00 at hour 42, 71.00 at hour 43, 73.00 at hour 44, 59.00 at hour 45, 64.00 at hour 46, 72.00 at hour 47, 72.00 at hour 48\n\nOxygen saturation:\n  100.00 at hour 5, 100.00 at hour 6, 97.50 at hour 8, 98.00 at hour 9, 98.00 at hour 10, 98.00 at hour 11, 98.00 at hour 12, 98.00 at hour 13, 99.00 at hour 14, 99.00 at hour 15, 99.00 at hour 16, 99.00 at hour 17, 99.00 at hour 18, 99.00 at hour 19, 100.00 at hour 20, 100.00 at hour 21, 100.00 at hour 22, 100.00 at hour 23, 100.00 at hour 24, 100.00 at hour 25, 100.00 at hour 26, 100.00 at hour 27, 100.00 at hour 28, 100.00 at hour 29, 100.00 at hour 30, 100.00 at hour 31, 100.00 at hour 32, 99.00 at hour 33, 100.00 at hour 34, 99.00 at hour 35, 99.00 at hour 36, 97.00 at hour 37, 97.00 at hour 38, 98.00 at hour 39, 98.00 at hour 40, 98.00 at hour 41, 98.00 at hour 42, 97.00 at hour 43, 98.00 at hour 44, 97.00 at hour 45, 97.00 at hour 46, 97.00 at hour 47, 98.00 at hour 48\n\npH:\n  7.38 at hour 5, 6.50 at hour 13, 7.40 at hour 34\n\nRespiratory rate:\n  17.67 at hour 5, 20.50 at hour 6, 22.00 at hour 7, 19.00 at hour 8, 24.00 at hour 9, 26.00 at hour 10, 25.00 at hour 11, 32.00 at hour 12, 30.00 at hour 13, 24.00 at hour 14, 26.00 at hour 15, 26.00 at hour 16, 27.00 at hour 17, 25.00 at hour 18, 26.00 at hour 19, 24.00 at hour 20, 26.00 at hour 21, 22.00 at hour 22, 25.00 at hour 25, 27.00 at hour 27, 26.00 at hour 28, 26.00 at hour 29, 23.00 at hour 30, 15.00 at hour 31, 16.00 at hour 32, 16.00 at hour 33, 16.00 at hour 34, 16.00 at hour 35, 21.00 at hour 36, 28.00 at hour 37, 20.00 at hour 38, 19.00 at hour 39, 19.00 at hour 40, 28.00 at hour 41, 28.00 at hour 42, 27.00 at hour 43, 24.00 at hour 44, 25.00 at hour 45, 19.00 at hour 46, 19.00 at hour 47, 19.00 at hour 48\n\nSystolic blood pressure:\n  97.00 at hour 5, 94.00 at hour 6, 143.00 at hour 7, 121.00 at hour 8, 122.00 at hour 9, 117.00 at hour 10, 103.00 at hour 11, 92.00 at hour 12, 108.00 at hour 13, 110.00 at hour 14, 103.00 at hour 15, 109.00 at hour 16, 109.00 at hour 17, 116.00 at hour 18, 111.00 at hour 19, 103.00 at hour 20, 105.00 at hour 21, 106.00 at hour 22, 128.00 at hour 23, 107.00 at hour 24, 117.00 at hour 25, 111.00 at hour 26, 116.00 at hour 27, 113.00 at hour 28, 100.00 at hour 29, 93.00 at hour 30, 106.00 at hour 31, 101.00 at hour 32, 113.00 at hour 33, 94.00 at hour 34, 105.00 at hour 35, 104.00 at hour 36, 105.00 at hour 37, 95.00 at hour 38, 117.00 at hour 39, 108.00 at hour 40, 127.00 at hour 41, 123.00 at hour 42, 109.00 at hour 43, 115.00 at hour 44, 97.00 at hour 45, 101.00 at hour 46, 112.00 at hour 47, 109.00 at hour 48\n\nTemperature:\n  37.00 at hour 4, 37.00 at hour 5, 37.22 at hour 8, 38.78 at hour 12, 38.22 at hour 14, 37.61 at hour 16, 36.83 at hour 20, 36.72 at hour 24, 37.22 at hour 28, 36.89 at hour 32, 37.00 at hour 36, 37.00 at hour 40, 37.67 at hour 44, 37.56 at hour 47\n\nWeight:\n  77.40 at hour 4, 75.70 at hour 38\n\nSimilar Patient Case:\n- Age: 77 years\n- Gender: Male \nPatient ID: SIMILAR_004\n\nPast Medical History:\n- Acute renal failure\n- Chronic kidney disease\n- Diabetes mellitus\n- Coronary atherosclerosis\n- Heart failure\n\nClinical Measurements (0–48h summary):\n- HR avg 115 bpm; frequent episodes >120 bpm\n- MAP frequently <65 mmHg despite vasopressors\n- SpO2 94-97% on FiO2 0.4-0.5\n- Glucose highly variable (150-330 mg/dL)\n- pH 7.38-7.48 (compensated)\n- GCS motor response predominantly 1 (no motor response)\n- Multiple organ dysfunction\n\nThis similar patient die in this ICU visit with In-hospital mortality = 1.\n</patient_data>",
        "data_source": "mimic_icu_mortality",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answers": ["Yes"],
                "patient_id": "10037861_34531557",
                "mortality": 1,
                "mortality_inunit": 1
            }
        }
    },
    {
        "text": "<question>\nBased on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.\n</question>\n\n<patient_data>\nPatient Demographics:\n- Age: 81 years\n- Gender: Female\n- Patient ID: 10037928_39804682\n\nPast Medical History:\n- Acute and unspecified renal failure\n- Anxiety disorders\n- Bacterial infection; unspecified site\n- Cancer of head and neck\n- Chronic kidney disease\n- Deficiency and other anemia\n- Diabetes mellitus with complications\n- Diabetes mellitus without complication\n- Disorders of lipid metabolism\n- Esophageal disorders\n- Essential hypertension\n- Fall\n- Fluid and electrolyte disorders\n- Gastrointestinal hemorrhage\n- Genitourinary symptoms and ill-defined conditions\n- Hypertension with complications and secondary hypertension\n- Inflammatory diseases of female pelvic organs\n- Menopausal disorders\n- Mood disorders\n- Other aftercare\n- Other diseases of bladder and urethra\n- Other fractures\n- Place of occurrence\n- Pneumonia (except that caused by tuberculosis or sexually transmitted disease)\n- Residual codes; unclassified\n- Urinary tract infections\n\nClinical Measurements (First 48 hours):\n\nDiastolic blood pressure:\n  62.00 at hour 3, 71.00 at hour 4, 61.00 at hour 5, 47.00 at hour 6, 69.00 at hour 7, 70.00 at hour 9, 51.00 at hour 10, 58.00 at hour 11, 90.00 at hour 12, 118.00 at hour 13, 65.00 at hour 14, 80.00 at hour 15, 65.00 at hour 16, 52.00 at hour 18, 62.00 at hour 19\n\nGlascow coma scale eye opening:\n  4.00 at hour 5, 4.00 at hour 15\n\nGlascow coma scale motor response:\n  6.00 at hour 5, 6.00 at hour 15\n\nGlascow coma scale verbal response:\n  5.00 at hour 5, 5.00 at hour 15\n\nGlucose:\n  419.00 at hour 1, 268.00 at hour 3, 241.50 at hour 4, 204.00 at hour 5, 157.00 at hour 6, 142.00 at hour 7, 230.00 at hour 9, 257.00 at hour 11, 296.00 at hour 16, 282.00 at hour 17, 276.00 at hour 18, 275.00 at hour 19\n\nHeart Rate:\n  108.00 at hour 3, 112.00 at hour 4, 111.00 at hour 5, 92.00 at hour 6, 89.00 at hour 7, 89.00 at hour 8, 92.00 at hour 9, 90.00 at hour 10, 91.00 at hour 11, 97.00 at hour 12, 93.00 at hour 13, 91.00 at hour 14, 94.00 at hour 15, 96.00 at hour 16, 97.00 at hour 17, 95.00 at hour 18, 94.00 at hour 19\n\nMean blood pressure:\n  74.50 at hour 3, 80.00 at hour 4, 79.00 at hour 5, 65.00 at hour 6, 80.00 at hour 7, 84.50 at hour 9, 71.00 at hour 10, 76.00 at hour 11, 102.00 at hour 12, 123.00 at hour 13, 79.00 at hour 14, 98.00 at hour 15, 69.00 at hour 16, 71.50 at hour 18, 79.00 at hour 19\n\nOxygen saturation:\n  97.00 at hour 1, 95.50 at hour 3, 95.50 at hour 4, 95.00 at hour 5, 95.00 at hour 6, 94.00 at hour 7, 95.00 at hour 8, 95.00 at hour 9, 96.00 at hour 10, 94.00 at hour 11, 95.00 at hour 12, 95.00 at hour 13, 94.00 at hour 14, 95.00 at hour 15, 94.00 at hour 16, 94.00 at hour 17, 95.00 at hour 18, 92.00 at hour 19\n\npH:\n  7.43 at hour 1, 7.37 at hour 11\n\nRespiratory rate:\n  22.00 at hour 3, 26.00 at hour 4, 27.00 at hour 5, 25.00 at hour 6, 21.00 at hour 7, 18.00 at hour 8, 23.00 at hour 9, 22.00 at hour 10, 27.00 at hour 11, 26.00 at hour 12, 26.00 at hour 13, 27.00 at hour 14, 24.00 at hour 15, 23.00 at hour 16, 24.00 at hour 17, 25.00 at hour 18, 27.00 at hour 19\n\nSystolic blood pressure:\n  121.50 at hour 3, 144.00 at hour 4, 132.00 at hour 5, 123.00 at hour 6, 123.00 at hour 7, 133.00 at hour 9, 131.00 at hour 10, 132.00 at hour 11, 147.00 at hour 12, 139.00 at hour 13, 128.00 at hour 14, 146.00 at hour 15, 102.00 at hour 16, 132.00 at hour 18, 129.00 at hour 19\n\nTemperature:\n  37.89 at hour 4, 37.28 at hour 6, 37.17 at hour 9, 36.56 at hour 14, 36.67 at hour 18\n\nWeight:\n  67.86 at hour 4\n\nSimilar Patient Case:\n- Age: 83 years.\n Gender: Female\n Patient ID: SIMILAR_005\n Past Medical History:  Essential hypertension, Diabetes mellitus,  Chronic kidney disease.\n Clinical Measurements (0–48h summary):  HR stable 90-100 bpm, MAP mostly 70-85 mmHg, SpO2 94-97% on room air/low FiO2, Glucose 140-250 mg/dL (improving with insulin), pH 7.37-7.43,GCS 15 (fully conscious), No vasopressor requirement\nThis similar patient survive in this ICU visit with In-hospital mortality = 0.\n</patient_data>",
        "data_source": "mimic_icu_mortality",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answers": ["No"],
                "patient_id": "10037928_39804682",
                "mortality": 0,
                "mortality_inunit": 0
            }
        }
    },
    {
        "text": "<question>\nBased on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.\n</question>\n\n<patient_data>\nPatient Demographics:\n- Age: 79 years\n- Gender: Male\n- Patient ID: 10037861_34531557\n\nPast Medical History:\n- Acute and unspecified renal failure\n- Bacterial infection; unspecified site\n- Cardiac dysrhythmias\n- Chronic kidney disease\n- Coagulation and hemorrhagic disorders\n- Congestive heart failure; nonhypertensive\n- Coronary atherosclerosis and other heart disease\n- Diabetes mellitus without complication\n- Disorders of lipid metabolism\n- Fluid and electrolyte disorders\n- Hypertension with complications and secondary hypertension\n- Other aftercare\n- Other circulatory disease\n- Pulmonary heart disease\n- Screening and history of mental health and substance abuse codes\n- Urinary tract infections\n\nClinical Measurements (First 48 hours):\n\nDiastolic blood pressure:\n  54.50 at hour 5, 54.00 at hour 6, 80.00 at hour 7, 61.00 at hour 8, 66.00 at hour 9, 58.00 at hour 10, 50.00 at hour 11, 47.00 at hour 12, 51.00 at hour 13, 53.00 at hour 14, 50.00 at hour 15, 51.00 at hour 16, 54.00 at hour 17, 55.00 at hour 18, 52.00 at hour 19, 54.00 at hour 20, 56.00 at hour 21, 54.00 at hour 22, 64.00 at hour 23, 55.00 at hour 24, 57.00 at hour 25, 55.00 at hour 26, 59.00 at hour 27, 56.00 at hour 28, 48.00 at hour 29, 47.00 at hour 30, 53.00 at hour 31, 50.00 at hour 32, 61.00 at hour 33, 50.00 at hour 34, 52.00 at hour 35, 50.00 at hour 36, 45.00 at hour 37, 46.00 at hour 38, 55.00 at hour 39, 53.00 at hour 40, 60.00 at hour 41, 58.00 at hour 42, 52.00 at hour 43, 52.00 at hour 44, 43.00 at hour 45, 45.00 at hour 46, 53.00 at hour 47, 54.00 at hour 48\n\nFraction inspired oxygen:\n  0.50 at hour 5, 0.40 at hour 7, 0.40 at hour 12, 0.40 at hour 13, 0.40 at hour 16, 0.40 at hour 19, 0.40 at hour 25, 0.40 at hour 29, 0.40 at hour 32, 0.30 at hour 36, 0.30 at hour 37, 0.30 at hour 40, 0.30 at hour 43, 0.30 at hour 48\n\nGlascow coma scale eye opening:\n  2.00 at hour 20, 2.00 at hour 22, 3.00 at hour 24, 3.00 at hour 26, 3.00 at hour 28, 3.00 at hour 30, 3.00 at hour 32, 3.00 at hour 34, 3.00 at hour 36, 3.00 at hour 38\n\nGlascow coma scale motor response:\n  1.00 at hour 5, 1.00 at hour 6, 1.00 at hour 7, 1.00 at hour 8, 1.00 at hour 9, 1.00 at hour 10, 1.00 at hour 11, 1.00 at hour 12, 1.00 at hour 13, 1.00 at hour 14, 1.00 at hour 15, 1.00 at hour 16, 1.00 at hour 17, 1.00 at hour 18, 6.00 at hour 20, 6.00 at hour 22, 6.00 at hour 24, 6.00 at hour 26, 6.00 at hour 28, 6.00 at hour 30, 6.00 at hour 32, 6.00 at hour 34, 6.00 at hour 36, 6.00 at hour 38, 6.00 at hour 41, 6.00 at hour 42, 6.00 at hour 44, 6.00 at hour 46, 6.00 at hour 48\n\nGlascow coma scale verbal response:\n  1.00 at hour 5, 1.00 at hour 6, 1.00 at hour 7, 1.00 at hour 8, 1.00 at hour 9, 1.00 at hour 10, 1.00 at hour 11, 1.00 at hour 12, 1.00 at hour 13, 1.00 at hour 14, 1.00 at hour 15, 1.00 at hour 16, 1.00 at hour 17, 1.00 at hour 18, 1.00 at hour 20, 1.00 at hour 22, 1.00 at hour 24, 1.00 at hour 26, 1.00 at hour 28, 1.00 at hour 30, 1.00 at hour 32, 1.00 at hour 34, 1.00 at hour 36, 1.00 at hour 38, 1.00 at hour 41, 1.00 at hour 42, 1.00 at hour 44, 1.00 at hour 46, 1.00 at hour 48\n\nGlucose:\n  166.00 at hour 5, 168.00 at hour 10, 165.00 at hour 18, 168.00 at hour 24, 204.00 at hour 30, 264.00 at hour 33, 217.00 at hour 42, 261.00 at hour 48\n\nHeart Rate:\n  67.00 at hour 5, 62.00 at hour 6, 81.00 at hour 7, 79.00 at hour 8, 79.00 at hour 9, 93.00 at hour 10, 83.00 at hour 11, 94.00 at hour 12, 93.00 at hour 13, 93.00 at hour 14, 91.00 at hour 15, 88.00 at hour 16, 85.00 at hour 17, 87.00 at hour 18, 92.00 at hour 19, 76.00 at hour 20, 88.00 at hour 21, 77.00 at hour 22, 84.00 at hour 23, 73.00 at hour 24, 85.00 at hour 25, 74.00 at hour 26, 80.00 at hour 27, 82.00 at hour 28, 100.00 at hour 29, 94.00 at hour 30, 84.00 at hour 31, 90.00 at hour 32, 84.00 at hour 33, 80.00 at hour 34, 82.00 at hour 35, 87.00 at hour 36, 81.00 at hour 37, 81.00 at hour 38, 84.00 at hour 39, 87.00 at hour 40, 89.00 at hour 41, 100.00 at hour 42, 81.00 at hour 43, 97.00 at hour 44, 102.00 at hour 45, 95.00 at hour 46, 93.00 at hour 47, 93.00 at hour 48\n\nMean blood pressure:\n  72.50 at hour 5, 68.50 at hour 6, 105.00 at hour 7, 81.00 at hour 8, 84.00 at hour 9, 77.00 at hour 10, 67.00 at hour 11, 61.00 at hour 12, 69.00 at hour 13, 72.00 at hour 14, 68.00 at hour 15, 71.00 at hour 16, 72.00 at hour 17, 76.00 at hour 18, 72.00 at hour 19, 70.00 at hour 20, 73.00 at hour 21, 71.00 at hour 22, 90.00 at hour 23, 73.00 at hour 24, 75.00 at hour 25, 73.00 at hour 26, 77.00 at hour 27, 75.00 at hour 28, 65.00 at hour 29, 62.00 at hour 30, 70.00 at hour 31, 68.00 at hour 32, 79.00 at hour 33, 65.00 at hour 34, 70.00 at hour 35, 68.00 at hour 36, 64.00 at hour 37, 62.00 at hour 38, 75.00 at hour 39, 70.00 at hour 40, 83.00 at hour 41, 81.00 at hour 42, 71.00 at hour 43, 73.00 at hour 44, 59.00 at hour 45, 64.00 at hour 46, 72.00 at hour 47, 72.00 at hour 48\n\nOxygen saturation:\n  100.00 at hour 5, 100.00 at hour 6, 97.50 at hour 8, 98.00 at hour 9, 98.00 at hour 10, 98.00 at hour 11, 98.00 at hour 12, 98.00 at hour 13, 99.00 at hour 14, 99.00 at hour 15, 99.00 at hour 16, 99.00 at hour 17, 99.00 at hour 18, 99.00 at hour 19, 100.00 at hour 20, 100.00 at hour 21, 100.00 at hour 22, 100.00 at hour 23, 100.00 at hour 24, 100.00 at hour 25, 100.00 at hour 26, 100.00 at hour 27, 100.00 at hour 28, 100.00 at hour 29, 100.00 at hour 30, 100.00 at hour 31, 100.00 at hour 32, 99.00 at hour 33, 100.00 at hour 34, 99.00 at hour 35, 99.00 at hour 36, 97.00 at hour 37, 97.00 at hour 38, 98.00 at hour 39, 98.00 at hour 40, 98.00 at hour 41, 98.00 at hour 42, 97.00 at hour 43, 98.00 at hour 44, 97.00 at hour 45, 97.00 at hour 46, 97.00 at hour 47, 98.00 at hour 48\n\npH:\n  7.38 at hour 5, 6.50 at hour 13, 7.40 at hour 34\n\nRespiratory rate:\n  17.67 at hour 5, 20.50 at hour 6, 22.00 at hour 7, 19.00 at hour 8, 24.00 at hour 9, 26.00 at hour 10, 25.00 at hour 11, 32.00 at hour 12, 30.00 at hour 13, 24.00 at hour 14, 26.00 at hour 15, 26.00 at hour 16, 27.00 at hour 17, 25.00 at hour 18, 26.00 at hour 19, 24.00 at hour 20, 26.00 at hour 21, 22.00 at hour 22, 25.00 at hour 25, 27.00 at hour 27, 26.00 at hour 28, 26.00 at hour 29, 23.00 at hour 30, 15.00 at hour 31, 16.00 at hour 32, 16.00 at hour 33, 16.00 at hour 34, 16.00 at hour 35, 21.00 at hour 36, 28.00 at hour 37, 20.00 at hour 38, 19.00 at hour 39, 19.00 at hour 40, 28.00 at hour 41, 28.00 at hour 42, 27.00 at hour 43, 24.00 at hour 44, 25.00 at hour 45, 19.00 at hour 46, 19.00 at hour 47, 19.00 at hour 48\n\nSystolic blood pressure:\n  97.00 at hour 5, 94.00 at hour 6, 143.00 at hour 7, 121.00 at hour 8, 122.00 at hour 9, 117.00 at hour 10, 103.00 at hour 11, 92.00 at hour 12, 108.00 at hour 13, 110.00 at hour 14, 103.00 at hour 15, 109.00 at hour 16, 109.00 at hour 17, 116.00 at hour 18, 111.00 at hour 19, 103.00 at hour 20, 105.00 at hour 21, 106.00 at hour 22, 128.00 at hour 23, 107.00 at hour 24, 117.00 at hour 25, 111.00 at hour 26, 116.00 at hour 27, 113.00 at hour 28, 100.00 at hour 29, 93.00 at hour 30, 106.00 at hour 31, 101.00 at hour 32, 113.00 at hour 33, 94.00 at hour 34, 105.00 at hour 35, 104.00 at hour 36, 105.00 at hour 37, 95.00 at hour 38, 117.00 at hour 39, 108.00 at hour 40, 127.00 at hour 41, 123.00 at hour 42, 109.00 at hour 43, 115.00 at hour 44, 97.00 at hour 45, 101.00 at hour 46, 112.00 at hour 47, 109.00 at hour 48\n\nTemperature:\n  37.00 at hour 4, 37.00 at hour 5, 37.22 at hour 8, 38.78 at hour 12, 38.22 at hour 14, 37.61 at hour 16, 36.83 at hour 20, 36.72 at hour 24, 37.22 at hour 28, 36.89 at hour 32, 37.00 at hour 36, 37.00 at hour 40, 37.67 at hour 44, 37.56 at hour 47\n\nWeight:\n  77.40 at hour 4, 75.70 at hour 38\n\nSimilar Patient Case:\n- Age: 77 years\n- Gender: Male \nPatient ID: SIMILAR_004\n\nPast Medical History:\n- Acute renal failure\n- Chronic kidney disease\n- Diabetes mellitus\n- Coronary atherosclerosis\n- Heart failure\n\nClinical Measurements (0–48h summary):\n- HR avg 115 bpm; frequent episodes >120 bpm\n- MAP frequently <65 mmHg despite vasopressors\n- SpO2 94-97% on FiO2 0.4-0.5\n- Glucose highly variable (150-330 mg/dL)\n- pH 7.38-7.48 (compensated)\n- GCS motor response predominantly 1 (no motor response)\n- Multiple organ dysfunction\n\nThis similar patient die in this ICU visit with In-hospital mortality = 1.\n</patient_data>",
        "data_source": "mimic_icu_mortality",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answers": ["Yes"],
                "patient_id": "10037861_34531557",
                "mortality": 1,
                "mortality_inunit": 1
            }
        }
    },
    {
        "text": "<question>\nBased on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.\n</question>\n\n<patient_data>\nPatient Demographics:\n- Age: 81 years\n- Gender: Female\n- Patient ID: 10037928_39804682\n\nPast Medical History:\n- Acute and unspecified renal failure\n- Anxiety disorders\n- Bacterial infection; unspecified site\n- Cancer of head and neck\n- Chronic kidney disease\n- Deficiency and other anemia\n- Diabetes mellitus with complications\n- Diabetes mellitus without complication\n- Disorders of lipid metabolism\n- Esophageal disorders\n- Essential hypertension\n- Fall\n- Fluid and electrolyte disorders\n- Gastrointestinal hemorrhage\n- Genitourinary symptoms and ill-defined conditions\n- Hypertension with complications and secondary hypertension\n- Inflammatory diseases of female pelvic organs\n- Menopausal disorders\n- Mood disorders\n- Other aftercare\n- Other diseases of bladder and urethra\n- Other fractures\n- Place of occurrence\n- Pneumonia (except that caused by tuberculosis or sexually transmitted disease)\n- Residual codes; unclassified\n- Urinary tract infections\n\nClinical Measurements (First 48 hours):\n\nDiastolic blood pressure:\n  62.00 at hour 3, 71.00 at hour 4, 61.00 at hour 5, 47.00 at hour 6, 69.00 at hour 7, 70.00 at hour 9, 51.00 at hour 10, 58.00 at hour 11, 90.00 at hour 12, 118.00 at hour 13, 65.00 at hour 14, 80.00 at hour 15, 65.00 at hour 16, 52.00 at hour 18, 62.00 at hour 19\n\nGlascow coma scale eye opening:\n  4.00 at hour 5, 4.00 at hour 15\n\nGlascow coma scale motor response:\n  6.00 at hour 5, 6.00 at hour 15\n\nGlascow coma scale verbal response:\n  5.00 at hour 5, 5.00 at hour 15\n\nGlucose:\n  419.00 at hour 1, 268.00 at hour 3, 241.50 at hour 4, 204.00 at hour 5, 157.00 at hour 6, 142.00 at hour 7, 230.00 at hour 9, 257.00 at hour 11, 296.00 at hour 16, 282.00 at hour 17, 276.00 at hour 18, 275.00 at hour 19\n\nHeart Rate:\n  108.00 at hour 3, 112.00 at hour 4, 111.00 at hour 5, 92.00 at hour 6, 89.00 at hour 7, 89.00 at hour 8, 92.00 at hour 9, 90.00 at hour 10, 91.00 at hour 11, 97.00 at hour 12, 93.00 at hour 13, 91.00 at hour 14, 94.00 at hour 15, 96.00 at hour 16, 97.00 at hour 17, 95.00 at hour 18, 94.00 at hour 19\n\nMean blood pressure:\n  74.50 at hour 3, 80.00 at hour 4, 79.00 at hour 5, 65.00 at hour 6, 80.00 at hour 7, 84.50 at hour 9, 71.00 at hour 10, 76.00 at hour 11, 102.00 at hour 12, 123.00 at hour 13, 79.00 at hour 14, 98.00 at hour 15, 69.00 at hour 16, 71.50 at hour 18, 79.00 at hour 19\n\nOxygen saturation:\n  97.00 at hour 1, 95.50 at hour 3, 95.50 at hour 4, 95.00 at hour 5, 95.00 at hour 6, 94.00 at hour 7, 95.00 at hour 8, 95.00 at hour 9, 96.00 at hour 10, 94.00 at hour 11, 95.00 at hour 12, 95.00 at hour 13, 94.00 at hour 14, 95.00 at hour 15, 94.00 at hour 16, 94.00 at hour 17, 95.00 at hour 18, 92.00 at hour 19\n\npH:\n  7.43 at hour 1, 7.37 at hour 11\n\nRespiratory rate:\n  22.00 at hour 3, 26.00 at hour 4, 27.00 at hour 5, 25.00 at hour 6, 21.00 at hour 7, 18.00 at hour 8, 23.00 at hour 9, 22.00 at hour 10, 27.00 at hour 11, 26.00 at hour 12, 26.00 at hour 13, 27.00 at hour 14, 24.00 at hour 15, 23.00 at hour 16, 24.00 at hour 17, 25.00 at hour 18, 27.00 at hour 19\n\nSystolic blood pressure:\n  121.50 at hour 3, 144.00 at hour 4, 132.00 at hour 5, 123.00 at hour 6, 123.00 at hour 7, 133.00 at hour 9, 131.00 at hour 10, 132.00 at hour 11, 147.00 at hour 12, 139.00 at hour 13, 128.00 at hour 14, 146.00 at hour 15, 102.00 at hour 16, 132.00 at hour 18, 129.00 at hour 19\n\nTemperature:\n  37.89 at hour 4, 37.28 at hour 6, 37.17 at hour 9, 36.56 at hour 14, 36.67 at hour 18\n\nWeight:\n  67.86 at hour 4\n\nSimilar Patient Case:\n- Age: 83 years.\n Gender: Female\n Patient ID: SIMILAR_005\n Past Medical History:  Essential hypertension, Diabetes mellitus,  Chronic kidney disease.\n Clinical Measurements (0–48h summary):  HR stable 90-100 bpm, MAP mostly 70-85 mmHg, SpO2 94-97% on room air/low FiO2, Glucose 140-250 mg/dL (improving with insulin), pH 7.37-7.43,GCS 15 (fully conscious), No vasopressor requirement\nThis similar patient survive in this ICU visit with In-hospital mortality = 0.\n</patient_data>",
        "data_source": "mimic_icu_mortality",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answers": ["No"],
                "patient_id": "10037928_39804682",
                "mortality": 0,
                "mortality_inunit": 0
            }
        }
    }
]
# Convert to the parquet format structure
dataset_entries = []

for i, item in enumerate(json_data):
    # Replace the question-only prompt with S3 format instructions
    original_text = item["text"]
    
    # Extract patient data from the original text
    patient_data_start = original_text.find("<patient_data>")
    patient_data_end = original_text.find("</patient_data>") + len("</patient_data>")
    patient_data = original_text[patient_data_start:patient_data_end]
    
    # Create new prompt with S3 format instructions
    s3_prompt = f"""You are a search copilot for medical literature retrieval. Based on an ICU patient's clinical data, you will help find relevant medical literature about mortality risk prediction.

You will go through a loop of <think> -> <query> -> <information> -> <think> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to gather relevant medical literature.

You should show your thinking process between <think> and </think>. You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched medical literature between <information> and </information>. You need to first think (<think>) on the retrieved information and put the doc id (1, 2, 3) of the important documents between <important_info> and </important_info> (e.g., <important_info>[1, 2]</important_info>).
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, formulate a new search query OR use <search_complete>False</search_complete> to indicate you want to continue searching. If you have sufficient information, use <search_complete>True</search_complete> to indicate that you have gathered enough information.

Focus on finding literature about:
1. ICU mortality prediction models and risk factors
2. Clinical parameters associated with ICU mortality
3. Specific conditions or abnormalities present in the patient
4. Prognostic indicators for critically ill patients

<question>
Based on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.
</question>

<patient_data>
[patient clinical information]
</patient_data>

The loop is as follows:
<think>
[analyze patient data and identify key risk factors]
</think>
<query>
{{
    "query": "[search query for medical literature]"
}} 
</query>
<information>
[top searched medical literature]
</information>
<think>
[analyze the search results for relevance to patient]
</think>
<important_info>
[doc ids of relevant papers]
</important_info>
<search_complete>
False
</search_complete>
<query>
{{
    "query": "[refined search query]"
}}
</query>
...... (several turns, max 4 turns in total)

<search_complete>
True
</search_complete>

Now, start the loop with the following patient data:
{patient_data}"""
    
    # Create the S3 format data entry
    data_entry = {
        "data_source": item["data_source"],
        "prompt": [{
            "role": "user",
            "content": s3_prompt
        }],
        "ability": "medical-reasoning",
        "reward_model": item["reward_model"],
        "extra_info": {
            'split': 'demo',
            'index': i,
            'patient_id': item["reward_model"]["ground_truth"]["patient_id"]
        }
    }
    dataset_entries.append(data_entry)

# Create DataFrame
df = pd.DataFrame(dataset_entries)

# Save to parquet file
output_file = '/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/demo_mimic_mortality.parquet'
df.to_parquet(output_file)

print(f"Saved {len(df)} entries to {output_file}")

# Print summary
print("\nDataFrame columns:")
print(df.columns.tolist())

print("\nFirst entry summary:")
print(f"- Data source: {df.iloc[0]['data_source']}")
print(f"- Patient ID: {df.iloc[0]['extra_info']['patient_id']}")
print(f"- Mortality: {df.iloc[0]['reward_model']['ground_truth']['mortality']}")
print(f"- Prompt content length: {len(df.iloc[0]['prompt'][0]['content'])} chars")

print("\nMortality distribution:")
for i, row in df.iterrows():
    mortality = row['reward_model']['ground_truth']['mortality']
    patient_id = row['reward_model']['ground_truth']['patient_id']
    print(f"  Patient {patient_id}: mortality={mortality}")


'''
def load_similar_data():
    data_path = '/scratch/bcew/ruikez2/intern/s3_med/data/patient_data_with_similar_dtw.json'
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        patients = json.load(f)
    
    print(f"Loaded {len(patients)} patients with similar cases")
    return patients

def format_patient_prompt(patient):
    """Format patient data into prompt text WITH similar case"""
    
    # Extract demographics
    demographics = patient['demographics']
    
    # Get past medical history - already in text format!
    past_medical_history = demographics.get('past_ccs_codes', [])
    
    if not past_medical_history:
        past_medical_history = ["No documented past medical history"]
    
    # Build the prompt text
    prompt_text = f"""<question>
Based on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.
</question>

<patient_data>
Patient Demographics:
- Age: {demographics['age']} years
- Gender: {demographics['gender']}
- Patient ID: {demographics['patient_id']}

Past Medical History:
"""
    
    for condition in past_medical_history[:20]:  # Limit to 20 conditions
        prompt_text += f"- {condition}\n"
    
    prompt_text += "\nClinical Measurements (First 48 hours):\n\n"
    
    # Add temporal data for each indicator
    temporal_data = patient['temporal_data']
    
    # Define the order of indicators to include
    indicator_order = [
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
    
    for indicator in indicator_order:
        if indicator in temporal_data and temporal_data[indicator] != "No data available":
            prompt_text += f"{indicator}:\n  {temporal_data[indicator]}\n\n"
    
    # Add similar patient case if available
    if patient.get('similar_patient_case') and patient['similar_patient_case'] != "No similar patient found":
        prompt_text += patient['similar_patient_case'] + "\n"
    
    prompt_text += "</patient_data>"
    
    return prompt_text

def main():
    # Load patient data with similar cases
    patients = load_similar_data()
    
    # Convert to the parquet format structure
    dataset_entries = []
    
    for i, patient in enumerate(patients):
        # Format the prompt with similar case
        prompt_text = format_patient_prompt(patient)
        
        # Create the S3 format data entry
        data_entry = {
            "data_source": "mimic_icu_mortality",
            "prompt": [{
                "role": "user",
                "content": prompt_text
            }],
            "ability": "medical-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answers": ["Yes"] if patient['mortality'] == 1 else ["No"],
                    "patient_id": patient['patient_id'],
                    "mortality": patient['mortality'],
                    "mortality_inunit": patient['mortality_inunit']
                }
            },
            "extra_info": {
                'split': 'train',
                'index': i,
                'patient_id': patient['patient_id']
            }
        }
        dataset_entries.append(data_entry)
    
    # Create DataFrame
    df = pd.DataFrame(dataset_entries)
    
    # Create output directory if it doesn't exist
    output_dir = '/scratch/bcew/ruikez2/intern/s3_med/data/mimic_mortality_temporal'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet file
    output_file = os.path.join(output_dir, 'mimic_mortality_with_similar.parquet')
    df.to_parquet(output_file)
    
    print(f"Saved {len(df)} entries to {output_file}")
    
    # Print summary
    print("\nDataFrame columns:")
    print(df.columns.tolist())
    
    print("\nFirst entry summary:")
    print(f"- Data source: {df.iloc[0]['data_source']}")
    print(f"- Patient ID: {df.iloc[0]['extra_info']['patient_id']}")
    print(f"- Mortality: {df.iloc[0]['reward_model']['ground_truth']['mortality']}")
    print(f"- Prompt content length: {len(df.iloc[0]['prompt'][0]['content'])} chars")
    
    print("\nMortality distribution:")
    mortality_counts = df['reward_model'].apply(lambda x: x['ground_truth']['mortality']).value_counts()
    print(f"  Survived (0): {mortality_counts.get(0, 0)}")
    print(f"  Died (1): {mortality_counts.get(1, 0)}")
    
    print("\nFirst prompt content (first 1500 chars):")
    print(df.iloc[0]['prompt'][0]['content'][:1500])
    print("...")

if __name__ == "__main__":
    main()


'''