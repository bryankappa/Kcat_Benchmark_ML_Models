from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class customSVMregressor:
    def __init__(self, **kwargs):
        """
        Initializes the SVR with the given keyword arguments.
        Any parameter supported by sklearn's SVR can be passed.
        """
        self.regressor = SVR(**kwargs)
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """Fit the model to the data."""
        # Scaling the input features
        X_scaled = self.scaler.fit_transform(X)
        self.regressor.fit(X_scaled, y)

    def predict(self, X):
        """Predict using the fitted model."""
        X_scaled  = self.scaler.transform(X)
        return self.regressor.predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate the model and return the Mean Squared Error."""
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)

    def set_params(self, **params):
        """Set new parameters for the regressor."""
        self.regressor.set_params(**params)

class SVMtuner:
# create a isntance class
    def __init__(self, svr):
        self.svr = svr
        self.best_params_ = None #storing variable to update the best_params_ when you add more tuning.

    def fit(self, X, y):
        # now set the best_params_

        C = [0.1, 1, 10]
        epsilon = [0.01, 0.1, 1]
        kernel = ['sigmoid', 'linear', 'rbf', 'poly']






    