import pandas as pd

from src.utility.FormulaParser import FormulaParser
from src.models.lm import lm

if __name__ == "__main__":
    model_formula = FormulaParser('dist~speed', ["speed", "dist"])

    df = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})
    print('Data to fit the model: \n', df)
    my_lm = lm("y~.-1", data=df)
    my_lm.fit()
    my_lm.summary()

    df1 = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})
    predict_df = pd.DataFrame({"x": [0, 1]})
    my_lm = lm("y~.", data=df1)
    my_lm.fit()
    y_hat = my_lm.predict(predict_df)
    print(y_hat)

