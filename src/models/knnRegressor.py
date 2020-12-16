from sklearn.neighbors import KNeighborsRegressor
from src.models.BaseRegressor import BaseRegressor


class knnRegressor(BaseRegressor):
    """KNN regressor wrapper for sklearn class KNeighborsRegressor.
    :formula: a string with the formula for y in terms of x_is
    :k: number of neighbors"""

    def __init__(self, formula, data=None, k = 3):
        # Initializing parent class
        super(knnRegressor, self).__init__(formula+"+0", data)
        self.sklearnModel = KNeighborsRegressor(n_neighbors = k)
        self.k = k

    def fit(self):
        self.sklearnModel.fit(self.X, self.y)

    def predict(self, newdata):
        X = self.getX(newdata)
        return self.sklearnModel.predict(X)
