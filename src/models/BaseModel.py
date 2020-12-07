from src.utility.helper import timing


class BaseModel:
    """Base class for all models in python like R"""

    def __init__(self):
        self.summary_text="" \
                        "\nCall: "\
                        "\nlm(formula = {formula}, data = cars) "\
                        "\n"\
                        "\nResiduals: "\
                        "\nMin\t1Q\tMedian\t3Q\tMax "\
                        "\n{resid_min}\t{resid_1Q}\t{resid_median}\t{resid_3Q}\t{resid_max}"\
                        "\n"\
                        "\nCoefficients:"\
                        "\n\t\tEstimate\tStd. Error\tt value\tPr(>|t|)"\
                        "\n{coef}"\
                        "\n---"\
                        "\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"\
                        "\n"\
                        "\nResidual standard error: {std_error} on {freedom} degrees of freedom"\
                        "\nMultiple R-squared:  {r_squared},	Adjusted R-squared:  {adjusted_r_squared}"\

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
        # return str

    def plot(self):
        raise NotImplementedError

    @timing
    def predict(self, newdata):
        raise NotImplementedError
