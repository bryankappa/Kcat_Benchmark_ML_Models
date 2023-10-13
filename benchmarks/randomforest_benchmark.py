from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

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

class RandomForestTuner:
    def __init__(self, rf):
        self.rf = rf
        self.best_params_ = None

    def fit(self, X, y):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4, 10]
        bootstrap = [True, False]
        
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        
        rf_random = RandomizedSearchCV(estimator=self.rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        rf_random.fit(X, y)
        self.best_params_ = rf_random.best_params_
        return self.best_params_
    
    def get_best_params(self):
        if self.best_params_ is None:
            raise ValueError("You need to fit the model first.")
        return self.best_params_
    
'''
rf = RandomForestClassifier()

# Initialize the tuner class
tuner = RandomForestTuner(rf)

# Fit the data and search for the best hyperparameters
best_params = tuner.fit(X_train, y_train)

# Retrieve the best parameters
print("Best parameters found: ", best_params)

# Or you could use the method to get the best parameters
print("Best parameters found: ", tuner.get_best_params())

'''