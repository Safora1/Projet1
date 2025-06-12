import pyshark
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fonction pour extraire un fichier PCAP et le convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    # Capture des paquets dans le fichier PCAP sans asynchrone
    cap = pyshark.FileCapture(pcap_file, use_json=True)
    data = []

    # Itérer à travers chaque paquet dans le fichier PCAP
    for pkt in cap:
        try:
            if 'IP' in pkt:
                # Extraire le protocole (ICMP, TCP, UDP, etc.)
                protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'UNKNOWN'

                # Déterminer si c'est une requête ou une réponse (basé sur ICMP)
                if 'ICMP' in pkt:
                    if hasattr(pkt.icmp, 'type'):
                        if pkt.icmp.type == '8':  # Type 8 est pour les requêtes Echo (ping request)
                            request_reply = 'request'
                        elif pkt.icmp.type == '0':  # Type 0 est pour les réponses Echo (ping reply)
                            request_reply = 'reply'
                        else:
                            request_reply = 'unknown'
                    else:
                        request_reply = 'unknown'
                else:
                    request_reply = 'unknown'

                row = {
                    'src_ip': pkt.ip.src,  # Adresse IP source
                    'dst_ip': pkt.ip.dst,  # Adresse IP destination
                    'protocol': protocol,  # Protocole (ICMP, TCP, UDP)
                    'length': int(pkt.length),  # Longueur du paquet
                    'time': str(pkt.sniff_time),  # Heure du paquet capturé
                    'ttl': int(pkt.ip.ttl) if hasattr(pkt.ip, 'ttl') else None,  # TTL
                    'id': int(pkt.icmp.id) if 'ICMP' in pkt else None,  # ID du paquet ICMP (si ICMP)
                    'seq': int(pkt.icmp.seq) if 'ICMP' in pkt else None,  # Séquence du paquet ICMP (si ICMP)
                    'request_reply': request_reply  # 'request' ou 'reply'
                }
                data.append(row)
        except AttributeError:
            continue  # Ignorer les paquets sans les attributs nécessaires

    return pd.DataFrame(data)


# Extraction des données
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Fichier PCAP normal
df_anormal = extract_pcap_to_dataframe('anomal.pcap')  # Fichier PCAP anormal

# Ajouter la colonne 'label' pour chaque ensemble de données
df_normal['label'] = 0  # 0 pour normal
df_anormal['label'] = 1  # 1 pour anormal

# Fusionner les deux datasets
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Nettoyage des données
# Remplacer les valeurs 'not_applicable' par NaN
df_final.replace('not_applicable', np.nan, inplace=True)

# Remplacer les NaN par la moyenne de chaque colonne numérique
df_final.fillna(df_final.mean(), inplace=True)

# Vérification des valeurs infinies
df_final.replace([np.inf, -np.inf], 0, inplace=True)

# Sélectionner les caractéristiques (features) et les labels
X = df_final[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]
y = df_final['label']  # La colonne 'label' indique si c'est normal (0) ou anormal (1)

# Encodage des variables catégorielles
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sauvegarder le dataset pré-traité dans un fichier CSV
df_final.to_csv('dataset_cleaned.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_cleaned.csv'.")
