from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from modeldnn  import *
from train import *
   # Predict on the validation set
y_pred_dnn = model_dnn .predict(valid[feature_columns]).flatten()

    # Calculate metrics
score_mae_dnn = mean_absolute_error(valid[target_column], y_pred_dnn)
score_rmse_dnn = math.sqrt(mean_squared_error(valid[target_column], y_pred_dnn))
score_r2_dnn = r2_score(valid[target_column], y_pred_dnn)

print(Fore.GREEN + 'DNN RMSE: {}'.format(score_rmse_dnn))
print(Fore.GREEN + 'DNN R^2 Score: {}'.format(score_r2_dnn))

    # Plot predictions
plt.figure(figsize=(15, 6))
plt.plot(valid['Date'], valid[target_column], label='Ground truth', color='orange')
plt.plot(valid['Date'], y_pred_dnn, label='DNN Forecast', color='green')
plt.title(f'DNN Prediction \n MAE: {score_mae_dnn:.2f}, RMSE: {score_rmse_dnn:.2f}, R^2 Score: {score_r2_dnn:.2f}', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NH Actual', fontsize=14)
plt.legend()
plt.show()