import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le dataset nettoyé
df = pd.read_csv('dataset_cleaned.csv')

# Étape 1 : Remplacer les valeurs 'not_applicable' et 'unknown' par NaN
df.replace(['not_applicable', 'unknown'], pd.NA, inplace=True)

# Étape 2 : Remplacer les NaN par la moyenne de chaque colonne numérique
df = df.apply(pd.to_numeric, errors='ignore')  # conversion si nécessaire
df.fillna(df.mean(numeric_only=True), inplace=True)

# Étape 3 : Remplacer les valeurs infinies
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Étape 4 : Sélection des caractéristiques et du label
X = df[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]
y = df['label']  # 0 pour normal, 1 pour anormal

# Étape 5 : Encodage des variables catégorielles
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Étape 6 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 7 : Création d'un DataFrame pour sauvegarde
df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
df_cleaned['label'] = y

# Étape 8 : Sauvegarde dans un fichier CSV
df_cleaned.to_csv('dataset_cleaned_final.csv', index=False)

# Étape 9 : Affichage clair de quelques lignes pour vérification
pd.set_option('display.max_columns', None)
print(df_cleaned.head())

print("✅ Données nettoyées, encodées et normalisées sauvegardées dans 'dataset_cleaned_final.csv'.")
