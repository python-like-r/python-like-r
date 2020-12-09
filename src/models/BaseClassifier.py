from src.models.BaseModel import BaseModel


class BaseClassifier(BaseModel):
    """
    TODO: Write comments
    """
    def __init__(self, formula, data=None):
        # Initializing parent class
        super(BaseClassifier, self).__init__(formula, data)

    def fit(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def predict(self, newdata):
        raise NotImplementedError
