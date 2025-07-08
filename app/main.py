# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.models.gesture_model import predict_gesture

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes. En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)

class SensorData(BaseModel):
    data: list  # Lista de 90 valores

@app.post("/predict")
def predict(data: SensorData):
    try:
        result = predict_gesture(data.data)
        return {"prediction": result}
    except ValueError as e:
        return {"error": str(e)}
