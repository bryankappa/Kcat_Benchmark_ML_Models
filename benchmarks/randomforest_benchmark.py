from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class CustomRandomForestRegressor:
    def __init__(self, **kwargs):
        """
        Initializes the RandomForestRegressor with the given keyword arguments.
        Any parameter supported by sklearn's RandomForestRegressor can be passed.
        """
        self.regressor = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        """Fit the model to the data."""
        self.regressor.fit(X, y)

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

# example calling random forest
# model = CustomRandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=4)
# model.fit(X_train, y_train)
# mse = model.evaluate(X_test, y_test)
# print(f"Mean Squared Error: {mse}")

# you can also change the parameters
# model.set_params(max_depth=10, min_samples_split=5)
