from fastapi import APIRouter
from app.models.gesture_model import predict_gesture
from app.schemas.gesture_schema import GestureInput

router = APIRouter()

@router.post("/predict")
def predict(data: GestureInput):
    result = predict_gesture(data)
    return {"prediction": result}
