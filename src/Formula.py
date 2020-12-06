from src.validate_descriptor import ValidFormula


def formula(attr):
    def decorator(cls):
        setattr(cls, attr, ValidFormula())
        return cls
    return decorator


@formula("formula")
class Formula:
    # Using Descriptors to validate Formula

    def __init__(self, formula_str):
        # check that column names only contain letters and underscores.
        self.formula = formula_str
