import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le dataset nettoyé
df = pd.read_csv('dataset_cleaned.csv')

# Étape 3 : Remplacer les valeurs 'not_applicable' par NaN, puis les traiter
df.replace('not_applicable', pd.NA, inplace=True)

# Étape 4 : Remplacer les NaN par la moyenne de chaque colonne numérique
df.fillna(df.mean(), inplace=True)

# Étape 5 : Vérifier et remplacer les valeurs infinies
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.fillna(df.mean(), inplace=True)

# Étape 6 : Sélectionner les caractéristiques (features) et les labels
X = df[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]  # Caractéristiques
y = df['label']  # Label (0 pour normal, 1 pour anormal)

# Étape 7 : Encodage des variables catégorielles (par exemple, IP source et destination, protocole)
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Étape 8 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Vérification des données après normalisation
print(X_scaled[:5])  # Afficher les premières lignes après normalisation

# Sauvegarder les données nettoyées dans un fichier CSV
df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)  # Créer un DataFrame avec les données normalisées
df_cleaned['label'] = y  # Ajouter la colonne 'label'
df_cleaned.to_csv('dataset_cleaned_final.csv', index=False)

print("Données nettoyées et normalisées sauvegardées dans 'dataset_cleaned_final.csv'.")
