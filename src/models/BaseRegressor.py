from src.models.BaseModel import BaseModel


class BaseRegressor(BaseModel):
    """
    Creating separate parent class for all Regressors inheriting from the BaseModel
    """
    def __init__(self, formula, data=None):
        # Initializing parent class
        super(BaseRegressor, self).__init__(formula, data)

