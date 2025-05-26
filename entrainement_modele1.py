import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from joblib import dump

# 1. Charger les données
df = pd.read_csv('dataset_final.csv')

# 2. Corriger les valeurs dans la colonne 'protocol'
df['protocol'] = df['protocol'].replace(-1, 2)  # Remplacer -1 par 2 pour 'autres'

# 3. Séparer les features (X) et la cible (y)
X = df.drop('label', axis=1)
y = df['label']

# 4. Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Sauvegarder le scaler pour la détection temps réel
dump(scaler, 'scaler.save')

# 6. Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Construire le modèle MLP
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 8. Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 9. Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 10. Sauvegarder le modèle entraîné
model.save('modele_ids.h5')

print("✅ Modèle entraîné et scaler sauvegardé avec succès.")
