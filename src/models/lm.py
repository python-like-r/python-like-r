import numpy as np

from src.models.BaseModel import BaseModel
from src.utility.FormulaParser import FormulaParser


class lm(BaseModel):
    """lm is used to fit linear models.
    :formula: a string of the form "y~x1+x2" (includes an intercept by default)
    :data: a pandas DataFrame
    """

    def __init__(self, formula='dist~speed', data=None):
        # Initializing parent class
        super(lm, self).__init__()

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
        self.p_values = np.full(len(self.coefs), 0)
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

    def rounded_str(self, num):
        return str(round(num, 3))

    def summary(self):
        intercept_idx = len(self.coefs)-1
        print(self.predictors)
        coef = 'Intercept'+'\t\t'+self.rounded_str(self.coefs[intercept_idx])+'\t\t'\
               +self.rounded_str(self.std_error[intercept_idx])+'\t\t'\
               +self.rounded_str(self.t_values[intercept_idx])+'\t\t'\
               +self.rounded_str(self.p_values[intercept_idx])+'\n'
        for i in range(len(self.coefs)-1):
            coef += self.predictors[i+1]+'\t\t'\
                    +self.rounded_str(self.coefs[i])+'\t\t'\
                    +self.rounded_str(self.std_error[i])+'\t\t'\
                    +self.rounded_str(self.t_values[i])+'\t\t'\
                    +self.rounded_str(self.p_values[i])+'\n'

        summary = self.summary_text.format(formula=self.formula, resid_min=min(self.residuals)
                                           , resid_1Q=np.quantile(self.residuals, .25)
                                           , resid_median=np.median(self.residuals)
                                           , resid_3Q=np.quantile(self.residuals, .75), resid_max=max(self.residuals)
                                           , coef=coef, std_error=self.std_error, r_squared=self.r_squared
                                           , adjusted_r_squared=self.adjusted_r_squared,freedom=None)
        print(summary)

    def plot(self):
        raise NotImplementedError
