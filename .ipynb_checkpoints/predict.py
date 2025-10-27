import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI(title = "Converted_Prediction")

@app.post("/predict")
def predict(lead: Lead):
    record = lead.dict()
    proba = pipeline.predict_proba([record])[0, 1]
    return {"conversion_probability": float(proba)}