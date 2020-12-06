from src.validate_descriptor import ValidFormula, ValidColumnNames


def formula(attr):
    def decorator(cls):
        setattr(cls, attr, ValidFormula())
        return cls
    return decorator


def predictors(attr):
    def decorator(cls):
        setattr(cls, attr, ValidColumnNames)
        return cls
    return decorator


@formula("formula")
@predictors("column_names")
class FormulaParser:
    # Using Descriptors to validate Formula

    def __init__(self, formula_str):
        # check that column names only contain letters and underscores.
        self.formula = formula_str
        self.predictors = self.get_predictors()
        self.response = self.get_response()

    def get_response(self):
        print('formula from get_response: ', self.formula)
        return 'dist'  # get the value from the formula string and set it

    def get_predictors(self):
        print('formula from get_predictors: ', self.formula)
        return ['speed']  # column_names ##get X and set here