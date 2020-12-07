import pandas as pd

from src.utility.FormulaParser import FormulaParser
from src.models.lm import lm

if __name__ == "__main__":
    model_formula = FormulaParser('dist~speed', ["speed", "dist"])
    # print('lm formula in main: ', model_formula.formula)
    # print('lm predictors in main: ', model_formula.predictors)
    # print('lm response in main: ', model_formula.response)

    df = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})
    # my_lm = lm("y~.-1", data=df)
    # my_lm = lm("y~.+0", data=df)
    my_lm = lm("y~.", data=df)
    my_lm.fit()
    print("coefs", my_lm.coefs)
    print("std errors:", my_lm.std_error)
    print('Summary: ', my_lm.summary())
