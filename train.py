import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données depuis un fichier ou une source de données
df = pd.read_excel('bmw.xlsx')
    # Set up feature columns and target column
feature_columns = [
        'NH Budget', 'CLIENT FORCAST S1',
        'Production Calendar', 'Customer Calendar',
        'Customer Consumption Last 12 week', 'Stock Plant : TIC Tool',
        'HC DIRECT', 'HC INDIRECT', 'ABS P', 'ABS NP', 'FLUCTUATION'
    ]
target_column = 'NH Actual'

    # Assume that 'WeekNumber' column represents the dates
multivariate_df = df[['Date', target_column] + feature_columns].copy()
multivariate_df.columns = ['ds', 'y'] + feature_columns

    # Train/validation split
train_size = int(0.85 * len(df))
train = df.iloc[:train_size]
valid = df.iloc[train_size:]

