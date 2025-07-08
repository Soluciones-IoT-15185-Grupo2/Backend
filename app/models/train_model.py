import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
data_path = BASE_DIR / "data" / "data_clean" / "dataset.csv"
OUTPUT_MODEL = BASE_DIR / "app" / "models" /'gesture_model.h5'

OUTPUT_CLASSES = BASE_DIR / "app" / "models" /'classes.npy'


df = pd.read_csv(data_path)

df = df.sort_values(by=['label', 'timestamp'])

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

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(60,)), 
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

model.save(OUTPUT_MODEL)
print(f"✅ Modelo guardado como {OUTPUT_MODEL}")

np.save(OUTPUT_CLASSES, encoder.classes_)
print("✅ Clases guardadas en app/models/classes.npy")
