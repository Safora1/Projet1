import pyshark
import pandas as pd
from keras.models import load_model
from joblib import load

# Charger le modèle entraîné
model = load_model('modele_ids.h5')

# Charger le scaler sauvegardé (tu dois l'avoir sauvegardé lors de l'entraînement)
scaler = load('scaler.save')

# Interface réseau à écouter (modifie selon ta configuration)
interface = 'enp0s3'  # remplace par eth0 ou autre si besoin

print(f"📡 Surveillance du trafic en cours sur : {interface}")

cap = pyshark.LiveCapture(interface=interface)

try:
    for pkt in cap.sniff_continuously():
        try:
            if 'IP' in pkt and pkt.transport_layer:
                proto = pkt.transport_layer
                length = int(pkt.length)

                # Encoder le protocole (0=TCP,1=UDP,2=autres)
                if proto == 'TCP':
                    proto_code = 0
                elif proto == 'UDP':
                    proto_code = 1
                else:
                    proto_code = 2

                # Préparer les données pour prédiction
                df = pd.DataFrame([[proto_code, length]], columns=['protocol', 'length'])

                # Normaliser avec le scaler chargé
                df_scaled = scaler.transform(df)

                # Prédiction
                prediction = model.predict(df_scaled)[0][0]

                if prediction > 0.5:
                    print(f"🚨 ALERTE ! Intrusion détectée (protocole={proto}, taille={length} octets)")
                else:
                    print(f"✅ Trafic normal (protocole={proto}, taille={length} octets)")
        except Exception:
            continue
except KeyboardInterrupt:
    print("\n🛑 Surveillance arrêtée par l'utilisateur.")
    cap.close()
