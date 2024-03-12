from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from traintest import *
from rophet import*

# Prédire sur l'ensemble de test
y_pred = model.predict(test.drop(columns=['y']))

# Calculer les métriques
score_mae = mean_absolute_error(test['y'], y_pred['yhat'])
score_rmse = mean_squared_error(test['y'], y_pred['yhat'], squared=False)
r_squared = r2_score(test['y'], y_pred['yhat'])

print(f'MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, R-squared: {r_squared:.2f}')

# Visualiser les résultats
plt.figure(figsize=(15, 6))
plt.plot(test['ds'], test['y'], label='Ground truth', color='orange')
plt.plot(test['ds'], y_pred['yhat'], label='Forecast', color='blue')
plt.title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, R-squared: {r_squared:.2f}', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NH Actual', fontsize=14)
plt.legend()
plt.show()

