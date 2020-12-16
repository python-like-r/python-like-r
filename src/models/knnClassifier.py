from sklearn.neighbors import KNeighborsClassifier
from src.models.BaseClassifier import BaseClassifier

class knnClassifier(BaseClassifier):
    """KNN Classifier wrapper for sklearn class KNeighborsClassifier. This is just to show how the class structure could be extended.
    :formula: a string with the formula for y in terms of x_is
    :k: number of neighbors"""

    def __init__(self, formula, data=None, k = 3):
        # Initializing parent class
        super(knnClassifier, self).__init__(formula+"+0", data)
        self.sklearnModel = KNeighborsClassifier(n_neighbors = k)
        self.k = k

    def fit(self):
        self.sklearnModel.fit(self.X, self.y)

    def predict(self, newdata):
        X = self.getX(newdata)
        return self.sklearnModel.predict(X)

    def predict_proba(self, newdata):
        X = self.getX(newdata)
        return self.sklearnModel.predict_proba(X)
