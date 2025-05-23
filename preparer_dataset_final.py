import pandas as pd

# 1. Charger les deux fichiers CSV
df_normal = pd.read_csv('trafic_normal.csv')
df_anormal = pd.read_csv('trafic_anormal.csv')

# 2. Ajouter une colonne 'label' : 0 = normal, 1 = anormal
df_normal['label'] = 0
df_anormal['label'] = 1

# 3. Supprimer les colonnes non utiles
colonnes_a_supprimer = ['src_ip', 'dst_ip', 'time']
df_normal = df_normal.drop(columns=colonnes_a_supprimer, errors='ignore')
df_anormal = df_anormal.drop(columns=colonnes_a_supprimer, errors='ignore')

# 4. Encodage du protocole (TCP, UDP, etc.) en valeur numérique
df_normal['protocol'] = df_normal['protocol'].astype('category').cat.codes
df_anormal['protocol'] = df_anormal['protocol'].astype('category').cat.codes

# 5. Fusionner les deux datasets
df_final = pd.concat([df_normal, df_anormal], ignore_index=True)

# 6. Sauvegarder dans un fichier final
df_final.to_csv('dataset_final.csv', index=False)

print("✅ Dataset final créé avec succès : dataset_final.csv")
