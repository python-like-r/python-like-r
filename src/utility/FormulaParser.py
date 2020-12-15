from src.utility.validate_descriptor import ValidFormula, ValidPredictors


def formula(attr):
    def decorator(cls):
        setattr(cls, attr, ValidFormula())
        return cls
    return decorator


def predictors(attr):
    def decorator(cls):
        setattr(cls, attr, ValidPredictors())
        return cls
    return decorator


@formula("formula")
@predictors("column_names")
class FormulaParser:
    # Using Descriptors to validate Formula

    def __init__(self, formula_str, columns):
        # check that column names only contain letters and underscores.
        self.formula = formula_str.replace(" ", "")
        self.columns = columns
        self.response = self.get_response()
        self.predictors = self.get_predictors()

    def get_response(self):
        response = self.formula.split("~")[0]
        if not (response in self.columns):
            raise ValueError("response not found in column names.")
        return response  # get the value from the formula string and set it

    def get_predictors(self):
        preds = self.formula.split("~")[1]
        # case: "y~."
        all_col = set(self.columns)
        all_col.remove(self.response)
        preds = preds.replace(".", "+".join(all_col))
        # split into list
        #handel cases like: y~.-1, y~.+-1
        preds_li = preds.replace("-", "+-").replace("++", "+").split("+")
        # make sets, also intercept cases
        # cases like: y~.+0, y~.-1, y~.-0
        preds_add = set([c for c in preds_li if c[0] != '-'])
        preds_remove = set([c[1:] for c in preds_li if c[0] == '-'])

        preds_add.add("Intercept")
        if "0" in preds_add:
            preds_add.remove("0")
            preds_remove.add("0")
        if "1" in preds_add:
            preds_add.remove("1")
        if "0" in preds_remove:
            preds_remove.remove("0")
            preds_remove.add("Intercept")
        if "1" in preds_remove:
            preds_remove.remove("1")
            preds_remove.add("Intercept")

        preds_set = preds_add.difference(preds_remove)

        return [col for col in ["Intercept"]+self.columns if col in preds_set]

    def has_intercept(self):
        return "Intercept" in self.predictors


# my_form = FormulaParser("dist~speed+wheels", ["dist", "speed", "wheels"])
# my_form.response
# my_form.predictors