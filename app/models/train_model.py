# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Cargar el dataset
data_path = os.path.join("..", "data", "data_clean", "combined_data.csv")  # Ajusta si el nombre cambia
df = pd.read_csv(data_path)

# Seleccionar solo columnas de acelerómetro y giroscopio
features = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
labels = df['gesture']  # Ajusta si el nombre de columna es diferente

# Codificar las etiquetas de texto a números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Normalizar los datos
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nPrecisión en el conjunto de prueba: {test_acc:.2f}')

# Guardar modelo
model.save("gesture_model.h5")
