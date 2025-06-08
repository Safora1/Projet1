import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from joblib import dump

# === 1. Charger le dataset ===
df = pd.read_csv('dataset_final.csv')

# === 2. Séparer les features et la cible ===
X = df.drop('label', axis=1)
y = df['label']

# === 3. Normaliser les données ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Sauvegarder le scaler pour la détection en temps réel ===
dump(scaler, 'scaler.save')

# === 5. Diviser les données ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 6. Construire le modèle MLP ===
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Classification binaire

# === 7. Compiler le modèle ===
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 8. Entraîner le modèle ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# === 9. Sauvegarder le modèle entraîné ===
model.save('modele_ids.h5')

print("✅ Modèle MLP entraîné et sauvegardé sous 'modele_ids.h5'")
print("✅ Scaler sauvegardé sous 'scaler.save'")
