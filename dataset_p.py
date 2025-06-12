import pyshark
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fonction pour extraire les données depuis un fichier PCAP et les convertir en DataFrame
def extract_pcap_to_dataframe(pcap_file):
    # Capture des paquets dans le fichier PCAP
    cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=True, disable_protocol="dns")
    data = []

    # Itérer à travers chaque paquet dans le fichier PCAP
    for pkt in cap:
        try:
            # Vérifier que le paquet contient un en-tête IP
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

    # Retourner le DataFrame avec les données extraites
    return pd.DataFrame(data)


# Étape 1 : Charger les données depuis les fichiers PCAP pour trafic normal et anormal
df_normal = extract_pcap_to_dataframe('normal.pcap')  # Fichier PCAP normal
df_anormal = extract_pcap_to_dataframe('anormal.pcap')  # Fichier PCAP anormal

# Étape 2 : Ajouter une colonne 'label' pour chaque ensemble de données
df_normal['label'] = 0  # 0 pour normal
df_anormal['label'] = 1  # 1 pour anormal

# Fusionner les deux datasets (trafic normal et anormal)
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# Étape 3 : Nettoyage des données
# Remplacer les valeurs 'not_applicable' par NaN, puis les traiter
df_final.replace('not_applicable', pd.NA, inplace=True)

# Remplacer les NaN par la moyenne de chaque colonne numérique
df_final.fillna(df_final.mean(), inplace=True)

# Vérifier et remplacer les valeurs infinies
df_final.replace([np.inf, -np.inf], 0, inplace=True)

# Étape 4 : Sélectionner les caractéristiques (features) et les labels
X = df_final[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]  # Les caractéristiques
y = df_final['label']  # La colonne 'label' indique si c'est normal (0) ou anormal (1)

# Étape 5 : Encodage des variables catégorielles (par exemple, IP source et destination)
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Étape 6 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 7 : Séparer les données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Étape 8 : Sauvegarder le dataset pré-traité dans un fichier CSV
df_final.to_csv('dataset_cleaned.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_cleaned.csv'.")
