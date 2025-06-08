import pandas as pd
from keras.models import load_model
from joblib import load

# === Charger scaler et modèle ===
scaler = load('scaler.save')
model = load_model('modele_ids.h5')

# === Charger le fichier à tester ===
df_test = pd.read_csv('test.csv')  # Le fichier doit avoir les colonnes: protocol, length

# === Appliquer la normalisation ===
X_scaled = scaler.transform(df_test)

# === Prédiction ===
y_pred = model.predict(X_scaled)
y_pred_labels = (y_pred > 0.5).astype('int32')

# === Afficher les résultats ===
for i, val in enumerate(y_pred_labels):
    print(f"Ligne {i+1} : {'🚨 Anomalie détectée' if val == 1 else '✅ Normal'} (score : {y_pred[i][0]:.4f})")
