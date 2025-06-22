from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="GloveTalk API")

app.include_router(router, prefix="/api", tags=["Prediction"])
