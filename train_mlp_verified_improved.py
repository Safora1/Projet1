import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Charger le dataset propre
df = pd.read_csv('dataset_cleaned_verified.csv')
print("✅ Données chargées.")

# 2. Séparer X et y
X = df.drop('label', axis=1)
y = df['label']

# 3. Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Calcul des pondérations de classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("📊 Pondérations appliquées :", class_weights)

# 5. Définir le modèle MLP amélioré
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))  # aide à éviter l'overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 6. Compiler
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 8. Entraîner le modèle avec class_weight
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# 9. Évaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Précision sur les données de test : {accuracy:.4f}")

# 10. Prédictions avec seuil ajusté
THRESHOLD = 0.6  # 🔁 à tester : 0.5, 0.6, 0.7
y_pred = (model.predict(X_test) > THRESHOLD).astype(int)

# 11. Rapport de classification
print("\n📊 Rapport de classification (seuil =", THRESHOLD, "):")
print(classification_report(y_test, y_pred))

# 12. Matrice de confusion
print("\n🧾 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 13. Sauvegarde du modèle
model.save('mlp_model_verified_improved.h5')
print("\n💾 Modèle sauvegardé sous 'mlp_model_verified_improved.h5'")

# 14. Tracer les courbes
plt.figure(figsize=(12, 5))

# Perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title("Courbe de perte")
plt.xlabel("Épochs")
plt.ylabel("Perte")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy entraînement')
plt.plot(history.history['val_accuracy'], label='Accuracy validation')
plt.title("Courbe d’accuracy")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
