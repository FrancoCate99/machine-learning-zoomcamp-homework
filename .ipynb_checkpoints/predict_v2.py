import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Cargar el pipeline
PIPELINE_PATH = "/code/pipeline_v2.bin"
with open(PIPELINE_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI(title="Converted_Prediction")

@app.post("/predict_v2")
def predict(lead: Lead):
    try:
        # Aquí reemplazás la predicción actual
        record = pd.DataFrame([{
            'lead_source': lead.lead_source, 
            'number_of_courses_viewed': lead.number_of_courses_viewed,
            'annual_income': lead.annual_income
        }])
        record['lead_source'] = record['lead_source'].astype('category')
        
        proba = pipeline.predict_proba(record)[0, 1]
        return {"conversion_probability": float(proba)}
    except Exception as e:
        return {"error": str(e)}