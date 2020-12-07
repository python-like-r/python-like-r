import numpy as np

from src.models import BaseModel
from src.FormulaParser import FormulaParser



class lm(BaseModel):
    """lm is used to fit linear models.
    :formula: a string of the form "y~x1+x2" (includes an intercept by default)
    :data: a pandas DataFrame
    """

    def __init__(self, formula='dist~speed', data=None):
        self.formula = formula
        self.data = data
        self.formula_parser = FormulaParser(formula, list(data.columns))

        # gets true y values
        self.response = self.formula_parser.response
        self.y = self.data[self.response]

        # gets design matrix
        self.predictors = self.formula_parser.predictors
        print(self.predictors)
        if self.formula_parser.has_intercept():
            self.X = self.data[[col for col in self.predictors if col != "Intercept"]]
            self.X["Intercept"] = 1
        else:
            self.X = self.data[self.predictors]

    def fit(self):
        # numpy array (n,k)
        X = np.array(self.X)
        n = X.shape[0]
        k = X.shape[1]
        # (n,)
        y = np.array(self.y)
        # (k,k)
        self.cov = np.linalg.inv(X.T.dot(X))
        # (k,)
        self.coefs = self.cov.dot(X.T).dot(y)
        # (n,)
        self.fitted_values = self.coefs.dot(X.T)
        # (n,)
        self.residuals = y - self.fitted_values
        # (1,)
        risidual_sum_of_squares = self.residuals.T.dot(self.residuals)
        # (1,)
        est_sigma_squared = risidual_sum_of_squares / (n - k)
        # (k,)
        self.std_error = np.diag(est_sigma_squared * self.cov) ** .5  # todo: 1d numpy array (k,)
        # (k,)
        self.t_values = self.coefs / self.std_error
        # (k,)
        self.p_values = None
        # (1,)
        self.r_squared = None
        # (1,)
        self.adjusted_r_squared = None

    #         np.quantile(self.re)

    def predict(self, newdata):
        # X dot coefs, ether (n,1) or (n,3)
        #         x =
        #         self.coefs.dot(X.T)
        pass

    def get_summary():
        # some code
        return some_string

    def summary(self):
        print(get_summary())

    def plot(self):
        raise NotImplementedError



