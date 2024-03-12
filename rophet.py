# Importer la fonction test depuis traintest.py
from traintest import *
from prophet import Prophet

# Créer un modèle Prophet
model = Prophet()

# Ajouter les régresseurs
for column in feature_columns:
    model.add_regressor(column)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(train)
