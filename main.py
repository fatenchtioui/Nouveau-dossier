import pandas as pd
from prepar import net
from prepar import matrix
from prepar import test_stationarity
import matplotlib.pyplot as plt
import seaborn as sns
# Import the visua function from the prepar module
from prepar import visua
from prepar import isnan
from prepar import sd
from statsmodels.tsa.stattools import adfuller 
from prepar import  resulat
from prepar import  visdef
from prepar import compose
from prepar import graph
import pandas as pd
from traintest import *
from rophet import *

from resultat import *
from resu import*
from modeldnn import model_dnn
from train import *
from resultat import evaluate_and_plot_predictions


# Read data from Excel file
df = pd.read_excel('Bmw_DATA.xlsx')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract week number, month, and year
df['WeekNumber'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.isocalendar().year

# Filter data for the years 2022 and 2023
df = df[(df['Year'] == 2022) | (df['Year'] == 2023)]

# Style the DataFrame
styled_df = df.head().style.set_properties(subset=['Date'], **{'background-color': 'dodgerblue'})

# Save the styled DataFrame as an HTML file
styled_df.to_html('styled_dataframe.html', render_links=True, escape=False)

print("Styled DataFrame saved as 'styled_dataframe.html'")
 
net(df) 
matrix(df)
window_size = int(input("Entrez la taille de la fenêtre pour la moyenne mobile : "))
timeseries = df['NH Actual']  # Extraction de la série temporelle du DataFrame
timeseries = timeseries.dropna()  # Suppression des valeurs NaN si nécessaire
test_stationarity(timeseries, window=window_size)
visua(df)
resulat(df)
visdef(df)
compose(df)
graph(df)
# Charger les données
isnan(df)
sd(df)

# Diviser les données en ensemble d'entraînement et de test
traintest()

# Entraîner le modèle et évaluer les résultats
model = model()

# Visualiser les résultats
resultat()





# Appeler la fonction test pour obtenir les données de test
train()

# Entraîner le modèle DNN
model_dnn=model_dnn()

# Évaluer et tracer les prédictions
resu()
