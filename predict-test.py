import requests

url = 'http://0.0.0.0:8080/predict'

patient = {
    "gender": "male", 
    "diabetes": "yes", 
    "family_history": "yes", 
    "smoking": "yes", 
    "obesity": "yes", 
    "alcohol_consumption": "yes", 
    "diet": "unhealthy", 
    "previous_heart_problems": "yes", 
    "medication_use": "yes", 
    "country": "united_states", 
    "age": 75, 
    "cholesterol": 360, 
    "heart_rate": 85, 
    "exercise_hours_per_week": 8, 
    "stress_level": 10, 
    "sedentary_hours_per_day": 4.987731820348275, 
    "income": 181290, 
    "bmi": 38, 
    "triglycerides": 369, 
    "physical_activity_days_per_week": 2, 
    "sleep_hours_per_day": 3, 
    "systolic_bp": 200, 
    "diastolic_bp": 130
}


request = requests.post(url, json=patient).json()

if request['attack risk']:
    print("Patient has a risk of heart attack.. Advise healthy lifestyle choices.")
else:
    print("Patient is not at risk of heart attack..")



