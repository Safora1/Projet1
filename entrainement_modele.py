import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model, save_model

# 1. Charger les données
df = pd.read_csv('dataset_final.csv')

# 2. Séparer les features (X) et la cible (y)
X = df.drop('label', axis=1)
y = df['label']

# 3. Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Créer le modèle Deep Learning
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 6. Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Entraîner
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 8. Sauvegarder le modèle
model.save('modele_ids.h5')
print("✅ Modèle entraîné et sauvegardé sous : modele_ids.h5")
