import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv('dataset_cleaned_verified.csv')
print("✅ Données chargées.")

# 2. Séparer X et y
X = df.drop('label', axis=1)
y = df['label']

# 3. Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Calcul des poids de classe
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}

# 5. Modèle MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. Entraînement
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# 8. Sauvegarde
model.save('mlp_model_verified_postfilter.h5')
print("💾 Modèle sauvegardé.")

# 9. Tester plusieurs seuils avec post-filtrage (condition sur 'length')
print("\n🔁 Test avec post-filtrage sur 'length' < 10")

for threshold in [0.5, 0.52, 0.54, 0.56]:
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > threshold).astype(int)

    # 🔒 Post-filtrage : si length < 10 → on annule l'alerte (0)
    if 'length' in X_test.columns:
        y_pred = np.where((y_pred == 1) & (X_test['length'] < 10), 0, y_pred)

    print(f"\n📊 Résultats pour seuil = {threshold} (avec post-filtrage)")
    print(classification_report(y_test, y_pred))
    print("🧾 Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
