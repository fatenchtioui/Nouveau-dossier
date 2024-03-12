import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données depuis un fichier ou une source de données
df = pd.read_excel('bmw.xlsx')

# Déclarer les colonnes caractéristiques et la colonne cible
feature_columns = [
    'NH Budget', 'CLIENT FORCAST S1',
    'Production Calendar', 'Customer Calendar',
    'Customer Consumption Last 12 week', 'Stock Plant : TIC Tool',
    'HC DIRECT', 'HC INDIRECT', 'ABS P', 'ABS NP', 'FLUCTUATION'
]
target_column = 'NH Actual'

# Calculer la taille de l'ensemble d'entraînement
train_size = int(0.85 * len(df))

# Créer un DataFrame contenant les colonnes sélectionnées
multivariate_df = df[['Date'] + [target_column] + feature_columns].copy()

# Renommer les colonnes pour correspondre à Prophet
multivariate_df.columns = ['ds', 'y'] + feature_columns

# Diviser le DataFrame en ensemble d'entraînement et de test
train, test = train_test_split(multivariate_df, test_size=0.15, shuffle=False)

# Enregistrer les ensembles d'entraînement et de test dans des fichiers CSV
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
