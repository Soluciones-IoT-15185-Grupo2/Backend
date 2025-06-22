from pydantic import BaseModel

class GestureInput(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
