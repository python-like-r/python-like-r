from src.validate import ValidColumnNames, ValidFormula


class Formula:
    # Using Descriptors to validate Column Names and Formula
    columns = ValidColumnNames()
    formula = ValidFormula()

    def __init__(self, formula_str, column_names):
        # check that column names only contain letters and underscores.
        self.formula = formula_str
        self.columns = column_names
