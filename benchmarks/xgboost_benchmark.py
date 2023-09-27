from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class customXGBregressor:
    def __init__(self, **kwargs):
        """
        Initializes the XGBregressor with the given keyword arguments.
        Any parameter supported by sklearn's RandomForestRegressor can be passed.

        """
        self.regressor = XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.regressor.fit(X,y)

    def predict(self, X):
        """Predict using the fitted model."""
        return self.regressor.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model and return the Mean Squared Error."""
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)

    def set_params(self, **params):
        """Set new parameters for the regressor."""
        self.regressor.set_params(**params)

