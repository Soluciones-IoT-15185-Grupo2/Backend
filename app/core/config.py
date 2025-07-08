import os
from dotenv import load_dotenv

load_dotenv()
# LOADING MODAL PATH
MODEL_PATH = os.getenv("MODEL_PATH", "models/gesture_model.h5")
