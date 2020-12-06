from src.models import BaseModel
from src.FormulaParser import FormulaParser

class lm(BaseModel):
    """lm is used to fit linear models.
    """

    def getX(self):

    def __init__(self, formula='dist~speed'):
        model_params = FormulaParser(formula)
        self.formula = model_params.formula
        self.predictors = model_params.predictors
        self.response = model_params.response

        # boolean for if we include an intercept in X
        # my_reg = lm(dist~.-1, data = cars) should handle the no intercept case
        self.intercept = self.formula.getIntercept()
        # gets desigin matrix
        self.X = self.formula.getX()
        # gets true y values
        self.Y = self.formula.getY()
        self.coefs = none

    def fit(self):
        # numpy array (k,)

        self.cov = np.linalg.inv(X.T.dot(X))  # todo: impliment (k,k)
        self.coefs = self.cov.dot(X.T).dot(Y)
        self.std_error = None  # todo: 1d numpy array (k,)
        self.fitted_values = None
        self.residuals = None  # todo: 1d numpy array (n,)

    def get_summary()
        # some code
        return some_string

    def summary(self):
        print(get_summary())

    def plot(self):
        raise NotImplementedError

    def predict(self, newdata):
        # X dot coefs, ether (n,1) or (n,3)
        self.fitted_values = None
