import pandas as pd
from sklearn.preprocessing import StandardScaler

# Étape 1 : Chargement du dataset brut
df = pd.read_csv('dataset_cleaned.csv')
print("✅ Dataset chargé avec succès.")

# Étape 2 : Remplacement des valeurs textuelles inutiles par NaN
df.replace(['not_applicable', 'unknown'], pd.NA, inplace=True)

# Étape 3 : Conversion des colonnes en numériques si nécessaire
df = df.apply(pd.to_numeric, errors='ignore')

# Étape 4 : Remplacement des NaN par la moyenne
df.fillna(df.mean(numeric_only=True), inplace=True)

# Étape 5 : Remplacement des valeurs infinies
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Étape 6 : Sélection des colonnes utiles
X = df[['src_ip', 'dst_ip', 'protocol', 'length', 'ttl', 'id', 'seq', 'request_reply']]
y = df['label']

# Étape 7 : Encodage des colonnes catégorielles
X = pd.get_dummies(X, columns=['src_ip', 'dst_ip', 'protocol', 'request_reply'], drop_first=True)

# Étape 8 : Suppression des colonnes constantes
X = X.loc[:, X.nunique() > 1]

# Étape 9 : Vérification des NaN restants
X.dropna(axis=1, inplace=True)

# Étape 10 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 11 : Création du DataFrame final
df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
df_cleaned['label'] = y.values

# Étape 12 : Vérification finale des classes
print("\n📊 Répartition des labels :")
print(df_cleaned['label'].value_counts())

# Étape 13 : Sauvegarde dans un nouveau fichier CSV
df_cleaned.to_csv('dataset_cleaned_verified.csv', index=False)
print("\n✅ Données nettoyées et normalisées sauvegardées dans 'dataset_cleaned_verified.csv'")
