import pandas as pd

from src.utility.FormulaParser import FormulaParser
from src.models.lm import lm
from src.models.knnRegressor import knnRegressor
from src.models.knnClassifier import knnClassifier

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

    my_knn = knnRegressor("y~.", data=df1)
    my_knn.fit()
    print(my_knn.predict(df1))

    df2 = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 1, 1]})
    my_knn_class = knnClassifier("y~.", data=df2)
    my_knn_class.fit()
    print(my_knn_class.predict_proba(df2))

