import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# === Configuración ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # sube de models -> app -> signs-backend
data_path = BASE_DIR / "data" / "data_clean" / "dataset.csv"
OUTPUT_MODEL = BASE_DIR / "app" / "models" /'gesture_model.h5'

OUTPUT_CLASSES = BASE_DIR / "app" / "models" /'classes.npy'


# === Paso 1: Cargar y ordenar el dataset ===
df = pd.read_csv(data_path)

# Asegura que los datos estén ordenados por tiempo para cada gesto
df = df.sort_values(by=['label', 'timestamp'])

# === Paso 2: Agrupar por gesto ===
# Cada gesto está compuesto por 10 registros (5 dedos por mano)
# Agrupamos cada 10 líneas consecutivas del mismo label
grouped = []
labels = []

for label in df['label'].unique():
    df_label = df[df['label'] == label]
    chunks = [df_label.iloc[i:i+10] for i in range(0, len(df_label), 10) if len(df_label.iloc[i:i+10]) == 10]
    for chunk in chunks:
        features = chunk[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.flatten()
        grouped.append(features)
        labels.append(label)

X = np.array(grouped)
y = np.array(labels)

# === Paso 3: Codificar etiquetas ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# === Paso 4: Dividir en entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# === Paso 5: Crear modelo ===
model = Sequential([
    Input(shape=(60,)),  # 10 sensores * 6 valores (ax,ay,az,gx,gy,gz)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Paso 6: Entrenar ===
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# === Paso 7: Guardar modelo ===
model.save(OUTPUT_MODEL)
print(f"✅ Modelo guardado como {OUTPUT_MODEL}")

# === Paso 8: Guardar las clases codificadas ===
np.save(OUTPUT_CLASSES, encoder.classes_)
print("✅ Clases guardadas en app/models/classes.npy")
