# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.models.gesture_model import predict_gesture

app = FastAPI()

class SensorData(BaseModel):
    data: list  # Lista de 90 valores

@app.post("/predict")
def predict(data: SensorData):
    try:
        result = predict_gesture(data.data)
        return {"prediction": result}
    except ValueError as e:
        return {"error": str(e)}
