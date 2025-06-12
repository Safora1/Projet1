import pyshark
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fonction pour extraire les données depuis un fichier PCAP
def extract_pcap_to_dataframe(pcap_file):
    cap = pyshark.FileCapture(pcap_file)
    data = []
    
    for pkt in cap:
        try:
            # Extraction des caractéristiques
            row = {
                'src_ip': pkt.ip.src if hasattr(pkt, 'ip') else None,
                'dst_ip': pkt.ip.dst if hasattr(pkt, 'ip') else None,
                'protocol': pkt.transport_layer if hasattr(pkt, 'transport_layer') else None,
                'length': int(pkt.length) if hasattr(pkt, 'length') else None,
                'ttl': int(pkt.ip.ttl) if hasattr(pkt, 'ip') else None,
                'id': pkt.ip.id if hasattr(pkt, 'ip') else None,
                'seq': pkt.tcp.seq if hasattr(pkt, 'tcp') else None,
                'request_reply': 'request' if 'Request' in str(pkt) else 'reply',
                'time': str(pkt.sniff_time)
            }
            data.append(row)
        except AttributeError:
            continue

    return pd.DataFrame(data)

# Étape 1 : Charger les données depuis les fichiers PCAP pour trafic normal et anormal
df_normal = extract_pcap_to_dataframe('normal.pcap')
df_anomal = extract_pcap_to_dataframe('anomal.pcap')

# Étape 2 : Ajouter une colonne 'label' pour chaque ensemble de données
df_normal['label'] = 0  # Trafic normal
df_anomal['label'] = 1  # Trafic anormal

# Fusionner les deux datasets (trafic normal et anormal)
df = pd.concat([df_normal, df_anomal], ignore_index=True)

# Afficher les premières lignes du dataset pour vérification
print(df.head())

# Étape 3 : Nettoyage des données
# Remplacer les valeurs 'not_applicable' par NaN, puis les traiter
df.replace('not_applicable', pd.NA, inplace=True)

# Remplacer les NaN par la moyenne de chaque colonne numérique
df.fillna(df.mean(), inplace=True)

# Vérifier et remplacer les valeurs infinies
df.replace([np.inf, -np.inf], 0, inplace=True)

# Étape 4 : Sélectionner les caractéristiques (features) et les labels
X = df[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]
y = df['label']  # La colonne 'label' indique si c'est normal (0) ou anormal (1)

# Étape 5 : Encodage des variables catégorielles (par exemple, IP source et destination)
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Étape 6 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 7 : Séparer les données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Afficher les données normalisées (facultatif)
print(X_scaled[:5])

# Étape 8 : Sauvegarder le dataset pré-traité dans un fichier CSV
df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
df_cleaned['label'] = y  # Ajouter la colonne 'label' pour chaque ligne
df_cleaned.to_csv('dataset_cleaned.csv', index=False)

print("Données traitées et sauvegardées dans 'dataset_cleaned.csv'.")
