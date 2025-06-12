import pyshark
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fonction pour extraire un fichier PCAP et le convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    cap = pyshark.FileCapture(pcap_file)
    data = []

    for pkt in cap:
        try:
            # Extraction des caractéristiques de chaque paquet
            row = {
                'src_ip': pkt.ip.src if hasattr(pkt, 'ip') else None,  # Adresse IP source
                'dst_ip': pkt.ip.dst if hasattr(pkt, 'ip') else None,  # Adresse IP destination
                'protocol': pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'UNKNOWN',  # Protocole (ICMP, TCP, UDP)
                'length': int(pkt.length) if hasattr(pkt, 'length') else None,  # Longueur du paquet
                'time': str(pkt.sniff_time),  # Heure du paquet capturé
                'ttl': int(pkt.ip.ttl) if hasattr(pkt, 'ip') else None,  # TTL
                'id': int(pkt.icmp.id) if 'ICMP' in pkt else None,  # ID du paquet ICMP (si ICMP)
                'seq': int(pkt.icmp.seq) if 'ICMP' in pkt else None,  # Séquence du paquet ICMP (si ICMP)
                'request_reply': 'request' if 'Request' in str(pkt) else 'reply' if 'Reply' in str(pkt) else 'unknown'  # Request/Reply basé sur ICMP
            }
            data.append(row)
        except AttributeError:
            continue  # Ignorer les paquets sans les attributs nécessaires

    return pd.DataFrame(data)

# Chargement des données PCAP pour trafic normal et anormal
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Remplacez par le chemin réel
df_anormal = extract_pcap_to_dataframe('anormal.pcap')  # Remplacez par le chemin réel

# Ajout de la colonne 'label' (0 pour normal, 1 pour anormal)
df_normal['label'] = 0  # Trafic normal
df_anormal['label'] = 1  # Trafic anormal

# Fusionner les deux datasets (normal et anormal)
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Nettoyage des données
df_final.replace('not_applicable', np.nan, inplace=True)  # Remplacer les valeurs 'not_applicable' par NaN
df_final.fillna(df_final.mean(), inplace=True)  # Remplacer les NaN par la moyenne de chaque colonne numérique

# Vérification des valeurs infinies et remplacement par 0
df_final.replace([np.inf, -np.inf], 0, inplace=True)

# Sélection des caractéristiques (features) et de la colonne 'label'
X = df_final[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]  # Features
y = df_final['label']  # Label (normal ou anormal)

# Encodage des variables catégorielles (par exemple, IP source et destination, protocole)
X_encoded = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Séparation des données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sauvegarder le dataset nettoyé dans un fichier CSV
df_final.to_csv('dataset_cleaned.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_cleaned.csv'.")
