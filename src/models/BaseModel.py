import numpy as np
import pandas as pd
from src.utility.FormulaParser import FormulaParser
from src.utility.helper import timing


class BaseModel:
    """
    Base class for all models in python like R
    Implementing base functionalities for all models
        1. Formula parsing
        2. Getting Predictors
    Defining methods that needs to be implemented by the child model classes inheriting this class
        1. fit
        2. predict
        3. plot
        4. summary
    """

    def __init__(self, formula, data):
        self.summary_text="" \
                        "\nCall: "\
                        "\nlm(formula = {formula}, data = {data}) "\
                        "\n"\
                        "\nResiduals: "\
                        "\nMin\t1Q\tMedian\t3Q\tMax "\
                        "\n{resid_min}\t{resid_1Q}\t{resid_median}\t{resid_3Q}\t{resid_max}"\
                        "\n"\
                        "\nCoefficients:"\
                        "\n\t\tEstimate\tStd. Error\tt value\t\tPr(>|t|)"\
                        "\n{coef}"\
                        "\n---"\
                        "\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"\
                        "\n"\
                        "\nResidual standard error: {std_error} on {freedom} degrees of freedom"\
                        "\nMultiple R-squared:  {r_squared},	Adjusted R-squared:  {adjusted_r_squared}"\

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
            X = X[self.predictors]
        else:
            X = data.loc[:,self.predictors]
        return X

    @timing
    def fit(self):
        raise NotImplementedError

    def summary(self):
        """
        this will return a string like
        TODO: write a decorator that can print instead of returning.
        the tests should have access to the return summary, the users should get the print summary.
        :return:
            Call:
            lm(formula = dist ~ ., data = cars)

            Residuals:
                Min      1Q  Median      3Q     Max
            -29.069  -9.525  -2.272   9.215  43.201

            Coefficients:
                        Estimate Std. Error t value Pr(>|t|)
            (Intercept) -17.5791     6.7584  -2.601   0.0123 *
            speed         3.9324     0.4155   9.464 1.49e-12 ***
            ---
            Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

            Residual standard error: 15.38 on 48 degrees of freedom
            Multiple R-squared:  0.6511,	Adjusted R-squared:  0.6438
            F-statistic: 89.57 on 1 and 48 DF,  p-value: 1.49e-12
        """
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    @timing
    def predict(self, newdata):
        raise NotImplementedError
