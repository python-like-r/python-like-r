import numpy as np
from scipy.stats import t
import statsmodels.api as sm
import matplotlib.pyplot as plt

from src.models.BaseModel import BaseModel
from src.utility.FormulaParser import FormulaParser
from src.utility.helper import rounded_str, get_p_significance


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
        self.X = self.getX(data)

    def getX(self, data):
        if self.formula_parser.has_intercept():
            X = data[[col for col in self.predictors if col != "Intercept"]]
            X["Intercept"] = 1
        else:
            X = data[self.predictors]
        return X

    def fit(self):
        # numpy array (n,k)
        X = np.array(self.X)
        n = X.shape[0]
        k = X.shape[1]
        self.n = n
        self.k = k
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
        # (n,)
        self.standardized_residuals = (self.residuals - np.mean(self.residuals)) / np.std(self.residuals)
        # (1,)
        residual_sum_of_squares = self.residuals.T.dot(self.residuals)
        # (1,)
        total_sum_of_squares = sum((y - np.mean(y)) ** 2)
        # (1,)
        self.est_sigma_squared = residual_sum_of_squares / (n - k)
        # (1,)
        self.residual_standard_error = np.sqrt(self.est_sigma_squared)
        # (k,)
        self.std_error = np.diag(self.est_sigma_squared * self.cov) ** .5  # todo: 1d numpy array (k,)
        # (k,)
        self.t_values = self.coefs / self.std_error
        # (k,)
        self.p_values = 2 * t.cdf(-np.abs(self.t_values), df=n - k)
        # (1,)
        self.r_squared = 1 - residual_sum_of_squares / total_sum_of_squares
        # (1,)
        self.adjusted_r_squared = 1 - ((n - 1) / (n - k)) * residual_sum_of_squares / total_sum_of_squares

    def predict(self, newdata, interval=None, level=.95):
        """predicts for the new data.
        :newdata: data that is being used for prediction
        :return: dot coefs, ether (n,1) or (n,3) with predictions
        """
        X = np.array(self.getX(newdata))
        y_hat = X.dot(self.coefs)
        if interval is None:
            return (y_hat)
        if interval in ["confidence", "conf", "c"]:
            #             student_t_conf = t.____(level, df = self.n-self.k)
            #             margin = student_t_conf * self.std_error
            raise NotImplementedError
        if interval in ["prediction", "pred", "p"]:
            raise NotImplementedError

    def plot(self, which=None):
        if which == 1:
            # Residuals vs Fitted
            plt.scatter(self.fitted_values, self.residuals)
            plt.plot([np.min(self.fitted_values), np.max(self.fitted_values)], [0, 0])
            plt.title("Residuals vs Fitted")
            plt.xlabel("Fitted\nlm(" + self.formula + ")")
            plt.ylabel("Residuals")
        if which == 2:
            # QQ plot
            fig = sm.qqplot(self.residuals)
            plt.plot([-1, 1], [-1, 1])
            plt.title("QQ plot")
            plt.xlabel("Theoretical quantiles\nlm(" + self.formula + ")")
            plt.ylabel("Standardized residuals")
        if which == 3:
            # Scale-Location
            y_val = np.sqrt(np.abs(self.standardized_residuals))
            plt.scatter(self.fitted_values, y_val)
            plt.ylim(-0.1, np.max(y_val) + 0.1)
            plt.title("Scale-Location")
            plt.xlabel("Fitted\nlm(" + self.formula + ")")
            plt.ylabel("sqrt(|Standardized residuals|)")
        if which == 4:
            raise NotImplementedError
        if which is None:
            raise NotImplementedError
#         fig, axs = plt.subplots(4)

    def summary(self):
        intercept_idx = len(self.coefs)-1
        pad_len = len(max(self.predictors, key=len))
        fill_char = ' '
        coef = ''
        pred_start_idx = 0
        coef_len = len(self.predictors)
        
        if self.formula_parser.has_intercept():
            coef = 'Intercept'.ljust(pad_len, fill_char)+'\t'\
                   +rounded_str(self.coefs[intercept_idx])+'\t\t'\
                   +rounded_str(self.std_error[intercept_idx])+'\t\t'\
                   +rounded_str(self.t_values[intercept_idx])+'\t\t'\
                   +rounded_str(self.p_values[intercept_idx])+get_p_significance(self.p_values[intercept_idx])+'\n'
            coef_len = len(self.predictors) - 1
            pred_start_idx = 1

        for i in range(coef_len):
            coef += self.predictors[i+pred_start_idx].ljust(pad_len, fill_char)+'\t'\
                    +rounded_str(self.coefs[i])+'\t\t'\
                    +rounded_str(self.std_error[i])+'\t\t'\
                    +rounded_str(self.t_values[i])+'\t\t'\
                    +rounded_str(self.p_values[i])+get_p_significance(self.p_values[i])+'\n'

        summary = self.summary_text.format(formula=self.formula, data=''
                                           , resid_min=min(self.residuals)
                                           , resid_1Q=np.quantile(self.residuals, .25)
                                           , resid_median=np.median(self.residuals)
                                           , resid_3Q=np.quantile(self.residuals, .75)
                                           , resid_max=max(self.residuals)
                                           , coef=coef
                                           , std_error=rounded_str(self.residual_standard_error)
                                           , r_squared=rounded_str(self.r_squared)
                                           , adjusted_r_squared=rounded_str(self.adjusted_r_squared)
                                           , freedom=self.n-self.k)
        print(summary)
