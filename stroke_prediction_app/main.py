from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd

model = joblib.load('./best_model.pkl')


class request_body(BaseModel):
    age: float
    hypertension: float
    heart_disease: float
    ever_married: float
    work_type: Literal['private', 'self-employed', 'children', 'govt_job', 'never_worked']
    avg_glucose_level: float
    bmi: float
    smoking_status: Literal['never_smoked', 'unknown', 'formerly_smoked', 'smokes']
    is_female: float
    lives_urban: float
    healthy_smoke_habit: float
    heart_diseases_history: float
    unhealthy_glucose_level: float
    age_bmi_percentile: float
    unhealthy_bmi_glucose: float
    
    
    
# sample test
data = {
    "age": 86.0,
    "hypertension": 0.0,
    "heart_disease": 0.0,
    "ever_married": 0.0,
    "work_type": 'private',
    "avg_glucose_level": 75.06,
    "bmi": 23.5,
    "smoking_status": 'never_smoked',
    "is_female": 1.0,
    "lives_urban": 1.0,
    "healthy_smoke_habit": 1.0,
    "heart_diseases_history": 0.0,
    "unhealthy_glucose_level": 0.0,
    "age_bmi_percentile": 46.0,
    "unhealthy_bmi_glucose": 0.0,
}
 
# Declaring our FastAPI instance
app = FastAPI()
 
# prediction endpoint
@app.post('/predict')
def predict(data : request_body):
    test_data = pd.DataFrame({
        'age': [data.age],
        'hypertension': [data.hypertension],
        'heart_disease': [data.heart_disease],
        'ever_married': [data.ever_married],
        'work_type': [data.work_type],
        'avg_glucose_level': [data.avg_glucose_level],
        'bmi': [data.bmi],
        'smoking_status': [data.smoking_status],
        'is_female': [data.is_female],
        'lives_urban': [data.lives_urban],
        'healthy_smoke_habit': [data.healthy_smoke_habit],
        'heart_diseases_history': [data.heart_diseases_history],
        'unhealthy_glucose_level': [data.unhealthy_glucose_level],
        'age_bmi_percentile': [data.age_bmi_percentile],
        'unhealthy_bmi_glucose': [data.unhealthy_bmi_glucose]
    })
    
    class_idx = model.predict(test_data)[0]
    
    return { 'stroke' : class_idx}