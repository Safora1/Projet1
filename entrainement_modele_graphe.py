import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# 1. Charger les données
df = pd.read_csv('dataset_cleaned_final.csv')

# 2. Séparer X et y
X = df.drop('label', axis=1)
y = df['label']

# 3. Division en données d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Définir le modèle MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # pour classification binaire

# 5. Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Early stopping (optionnel mais recommandé)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. Entraîner le modèle
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 8. Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Précision sur les données de test : {accuracy:.4f}")

# 9. Prédictions et rapport
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# 10. Matrice de confusion (optionnel)
print("🧾 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 11. Sauvegarder le modèle
model.save('mlp_model.h5')
print("\n💾 Modèle sauvegardé sous le nom 'mlp_model.h5'")

# 12. Tracer l’évolution des performances
plt.figure(figsize=(12, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title("Courbe de perte")
plt.xlabel("Épochs")
plt.ylabel("Perte")
plt.legend()

# Courbe d’accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy entraînement')
plt.plot(history.history['val_accuracy'], label='Accuracy validation')
plt.title("Courbe d’accuracy")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
