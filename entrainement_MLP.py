import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Charger le dataset depuis le fichier CSV
df = pd.read_csv('dataset_final_with_request_reply.csv')

# Séparer les caractéristiques (features) et les labels
X = df[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]
y = df['label']

# Convertir les caractéristiques catégorielles en numériques (encoding)
X = pd.get_dummies(X)

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser le dataset en données d'entraînement et de test (80% pour l'entraînement, 20% pour le test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer un modèle MLP avec Keras
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Couche d'entrée et première couche cachée
model.add(Dense(32, activation='relu'))  # Deuxième couche cachée
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie avec activation sigmoïde pour la classification binaire (normal vs anormal)

# Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy du modèle : {accuracy:.4f}")

# Sauvegarder le modèle en format .h5
model.save('mlp_model.h5')
print("Modèle MLP sauvegardé sous 'mlp_model.h5'.")
