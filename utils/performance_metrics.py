from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('MSE: ', mse)
    print('R2:', r2_score(y_test, y_pred))
    print('RMSE:', np.sqrt(mse))
    print("Pearson Correlation:", np.corrcoef(y_test, y_pred)[0,1])


