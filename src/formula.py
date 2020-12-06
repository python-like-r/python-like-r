from src.validate import ValidColumnNames, ValidFormula


class Formula:
    # these functions check the list of column names
    columns = ValidColumnNames()
    formula = ValidFormula()

    def __init__(self, column_names, formula_str):
        # check that column names only contain letters and underscores.
        self.columns = column_names
        self.formula = formula_str
