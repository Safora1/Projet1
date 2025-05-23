import pyshark
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Charger le modèle entraîné
model = load_model('modele_ids.h5')

# Initialiser un scaler (attention : idéalement tu devrais sauvegarder le scaler entraîné)
scaler = StandardScaler()

# Interface réseau à écouter (vérifie que c'est eth0 ou modifie selon ta topologie GNS3)
interface = 'eth0'
cap = pyshark.LiveCapture(interface=interface)

print("📡 Surveillance du trafic en cours sur :", interface)

for pkt in cap.sniff_continuously():
    try:
        if 'IP' in pkt and pkt.transport_layer:
            proto = pkt.transport_layer
            length = int(pkt.length)

            # Encoder le protocole comme dans l'entraînement (0 = TCP, 1 = UDP, 2 = autre)
            if proto == 'TCP':
                proto_code = 0
            elif proto == 'UDP':
                proto_code = 1
            else:
                proto_code = 2

            # Créer une ligne de données
            df = pd.DataFrame([[proto_code, length]], columns=['protocol', 'length'])

            # Appliquer le même type de normalisation
            df_scaled = scaler.fit_transform(df)  # ⚠️ Dans un vrai projet : utiliser le scaler appris

            # Prédiction du modèle
            prediction = model.predict(df_scaled)[0][0]

            if prediction > 0.5:
                print(f"🚨 ALERTE ! Intrusion détectée (protocole={proto}, taille={length} octets)")
            else:
                print(f"✅ Trafic normal (protocole={proto}, taille={length} octets)")

    except Exception as e:
        continue
