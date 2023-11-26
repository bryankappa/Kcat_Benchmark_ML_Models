from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

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

    def set_params(self, **params):
        """Set new parameters for the regressor."""
        self.regressor.set_params(**params)

class customLGBMRegressor:
    def __init__(self, **kwargs):
        """
        Initializes the LGBMRegressor with the given keyword arguments.
        Any parameter supported by LGBMRegressor can be passed.

        """
        self.regressor = LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        """Predict using the fitted model."""
        return self.regressor.predict(X)

    def set_params(self, **params):
        """Set new parameters for the regressor."""
        self.regressor.set_params(**params)


class customCatBoostRegressor:
    def __init__(self, **kwargs):
        """
        Initializes the CatBoostRegressor with the given keyword arguments.
        Any parameter supported by LGBMRegressor can be passed.

        """       
        self.regressor = CatBoostRegressor(**kwargs)
    
    def fit(self, X, y):
        """ fitted model. """
        self.regressor.fit(X, y)

    def predict(self, X):
        """ Predict on the fitted model"""
        return self.regressor.predict(X)
    
    def set_params(self, **params):
        """Set new parameters for the regressor."""
        self.regressor.set_params(**params)

# setting up the hyperparamter tuning for Boosting models.

# class BoostTuner:

#     def __init__(self, boost):
#         self.boost = boost
#         self.best_params_= None # set to none to be able to add to the best parameters.
    
#     def fit(self, X, y):
#         self.best_params_ = 

        


    


    




