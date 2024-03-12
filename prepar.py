
import pandas as pd
def read_data():
# Read the Excel file into a DataFrame
  df = pd.read_excel('Bmw_DATA.xlsx')

# Display the first 5 rows of the DataFrame
  print(df.head(5))
from datetime import datetime, date
import pandas as pd

def process_dates(df):
    # Convert 'Date' to datetime
    df.Date = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Extract week number, month, and year
    df.WeekNumber = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.isocalendar().year

    # Filter data for the years 2022 and 2023
    df = df[(df['Year'] == 2022) | (df['Year'] == 2023)]

    # Print the first few rows with formatted Date column
    print(df.head().style.set_properties(subset=['Date'], **{'background-color': 'dodgerblue'}))

    # Optionally, you can return the modified DataFrame
    return df

# Example usage:
# df = process_dates(df)



def net(df):
    columns_to_drop = ['Sales Bud', 'Sales Act ',' Sales Actual/Budget', 'NH Actual/Budget', 'Efficiency Bud ','Efficiency Act', 'Efficiency Actual/Budget']

    # Dropping the columns
    df.drop(columns=columns_to_drop, inplace=True)
    print(df.info())
    
    # Fill missing values with zeros
    df.fillna(0, inplace=True)
    
    # Print the DataFrame description
    print(df.describe())
    
    return df


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def matrix(df):
  correlation_matrix =df.corr()
  mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Afficher la heatmap de la matrice de corrélation en utilisant le masque
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
  plt.title("Triangle inférieur de la matrice de corrélation")
  plt.show()
  correlation_matrix.to_csv('correlation_matrix.csv', index=False)
  return(correlation_matrix)

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Test de Dickey-Fuller augmenté pour la stationnarité
def test_stationarity(timeseries, window):
    # Calcul de la moyenne mobile et de l'écart-type mobile
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Affichage des statistiques mobiles
    orig = plt.plot(timeseries, color='blue', label='Données originales')
    mean = plt.plot(rolmean, color='red', label='Moyenne mobile')
    std = plt.plot(rolstd, color='black', label='Écart-type mobile')
    plt.legend(loc='best')
    plt.title('Moyenne mobile et écart-type mobile')
    plt.show()

    # Test de Dickey-Fuller augmenté
    print('Résultats du test de Dickey-Fuller augmenté :')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Statistique du test', 'Valeur p', '#Lags utilisés', 'Nombre d\'observations utilisées'])
    for key, value in dftest[4].items():
        dfoutput['Valeur critique (%s)' % key] = value
    print(dfoutput)

def visua(df):
      # Créer la figure et les axes
    fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, figsize=(16, 12))

# Tracer la série temporelle
    sns.lineplot(x=df['Date'], y=df['NH Actual'], color='dodgerblue', ax=ax)

# Titre du graphique
    ax.set_title('NH Actual Volume', fontsize=14)

# Définir les limites de l'axe x en fonction des données
    ax.set_xlim(df['Date'].min(), df['Date'].max())

# Afficher le graphique

    return(plt.show())
from statsmodels.tsa.stattools import adfuller 
def resulat(df):
   result = adfuller(df['NH Actual'].values)
   return(result)
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def visualize_adfuller_results(series, title, ax):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen'
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'orange'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'red'
    else:
        linecolor = 'purple'

    sns.lineplot(x=range(len(series)), y=series, ax=ax, color=linecolor)
    ax.set_title(f'ADF Statistic: {adf_stat:0.3f}\n'
                 f'p-value: {p_val:0.3f}\n'
                 f'Critical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)

def visdef(df):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
    visualize_adfuller_results(df['NH Actual'].values, 'NH Actual', ax)
    plt.tight_layout()
    plt.show()

# Example usage:
# visdef(df)
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def compose(df):
    # Assuming df is your DataFrame with a datetime index
    result = seasonal_decompose(df['NH Actual'], model='additive', period=4)

    # Plot the decomposition components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

    df['NH Actual'].plot(ax=ax1, title='Original Time Series')
    result.trend.plot(ax=ax2, title='Trend Component')
    result.seasonal.plot(ax=ax3, title='Seasonal Component')
    result.resid.plot(ax=ax4, title='Residual Component')

    plt.tight_layout()
    plt.show()

# Example usage:
# compose(df)
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def graph(df):
    # Supposons que df['NH Actual'] est un DataFrame, extrayons la colonne qui nous intéresse
    nh_actual_series = df['NH Actual']

    # Vérifions la forme de la série temporelle
    print(nh_actual_series.shape)

    # Si la série a une dimension supplémentaire, nous pouvons la transformer en une série unidimensionnelle
    nh_actual_series = nh_actual_series.squeeze()

    # Vérifions à nouveau la forme pour confirmer qu'elle est unidimensionnelle
    print(nh_actual_series.shape)

    # Tracer l'autocorrélation et l'autocorrélation partielle avec moins de lags
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    plot_acf(nh_actual_series, lags=99, ax=ax[0])
    plot_pacf(nh_actual_series, lags=50, ax=ax[1])
    plt.show()

# Example usage:
# graph(df)
def isnan(df):
    df.fillna(0, inplace=True)
    return df
# Example usage:
def sd(df):
    df.to_excel('bmw.xlsx', index=False)
    return df.isnull().sum().max()
