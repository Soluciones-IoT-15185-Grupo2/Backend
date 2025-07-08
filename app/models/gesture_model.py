# app/models/gesture_model.py

import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model("app/models/gesture_model.h5")
classes = np.load("app/models/classes.npy")

def predict_gesture(sensor_array: list) -> str:
    """Recibe una lista de 60 valores (ax, ay, az, gx, gy, gz) de ambos guantes"""
    if len(sensor_array) != 60:
        raise ValueError("Se esperaban 60 valores de entrada")
    X = np.array(sensor_array).reshape(1, -1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    prediction = model.predict(X)
    predicted_label = classes[np.argmax(prediction)]
    return predicted_label
