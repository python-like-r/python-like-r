import Formula

class lm(model):
    """lm is used to fit linear models.
    """

    def getX(self):

    def __init__(self, call):
        self.formula = Formula(call)
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
        self.coefs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        self.cov = None  # todo: impliment (k,k)
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
