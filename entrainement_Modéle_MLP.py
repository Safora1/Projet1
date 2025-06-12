# Entrainement_MLP.py

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Étape 1 : Charger les données pré-traitées
df = pd.read_csv('dataset_cleaned.csv')

# Séparer les caractéristiques (features) et les labels
X = df.drop(columns=['label'])  # Enlever la colonne 'label'
y = df['label']  # La colonne 'label' indique si c'est normal (0) ou anormal (1)

# Étape 2 : Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 3 : Séparer le dataset en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Étape 4 : Création du modèle MLP
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Couche d'entrée
model.add(Dense(32, activation='relu'))  # Couche cachée
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie pour classification binaire (normal/anormal)

# Étape 5 : Compiler le modèle
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Étape 6 : Entraîner le modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Étape 7 : Évaluation du modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy du modèle : {accuracy:.4f}")

# Sauvegarder le modèle entraîné
model.save('mlp_model.h5')
print("Modèle MLP sauvegardé sous 'mlp_model.h5'.")

