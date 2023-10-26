import pickle

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

patient = {
    'gender': 'female', 
    'diabetes': 'no', 
    'family_history': 'no', 
    'smoking': 'no', 
    'obesity': 'yes', 
    'alcohol_consumption': 'no', 
    'diet': 'average', 
    'previous_heart_problems': 'yes', 
    'medication_use': 'yes', 
    'country': 'united_states', 
    'age': 37, 
    'cholesterol': 360, 
    'heart_rate': 60, 
    'exercise_hours_per_week': 17.160756256752286, 
    'stress_level': 10, 
    'sedentary_hours_per_day': 4.987731820348275, 
    'income': 181290, 
    'bmi': 35.661224922332416, 
    'triglycerides': 369, 
    'physical_activity_days_per_week': 5, 
    'sleep_hours_per_day': 5, 
    'systolic_bp': 172, 
    'diastolic_bp': 71
    }


X = dv.transform([patient])
y_pred = model.predict_proba(X)[0, 1]

print(f"Patient: \n{patient}")
print(f"\nHeart Attack Risk Probability {round(y_pred, 3)}")