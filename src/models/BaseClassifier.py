from src.models.BaseModel import BaseModel


class BaseClassifier(BaseModel):
    """
    Base classifier implements `predict_proba` in addition to `fit`, `predit` which is inherited from `BaseModel`
    """
    def __init__(self, formula, data=None):
        # Initializing parent class
        super(BaseClassifier, self).__init__(formula, data)

    def predict_proba(self, newdata):
        raise NotImplementedError
