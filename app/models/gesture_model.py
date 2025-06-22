import numpy as np
import tensorflow as tf
from app.core.config import MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)
label_map = ['a', 'b', 'c', 'd', 'e']  # Reemplaza con tus clases reales

def predict_gesture(data):
    input_array = np.array([[data.ax, data.ay, data.az, data.gx, data.gy, data.gz]])
    input_array = input_array / np.linalg.norm(input_array)
    prediction = model.predict(input_array)
    predicted_index = np.argmax(prediction)
    return label_map[predicted_index]
