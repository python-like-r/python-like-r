import pandas as pd

from unittest import TestCase
from src.utility.FormulaParser import FormulaParser
from src.models.BaseModel import BaseModel
from src.models.BaseRegressor import BaseRegressor
from src.models.BaseClassifier import BaseClassifier
from src.models.lm import lm

class FormulaParserTest(TestCase):

    def test_init(self):
        #test invalid chars
        self.assertRaises(ValueError, lambda: FormulaParser('y~x%', ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('y~x!', ["x", "y"]))
        #test number of "~" is not 1
        self.assertRaises(ValueError, lambda: FormulaParser('y~~x',  ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('yx', ["x", "y"]))
        #test list properties
        self.assertRaises(ValueError, lambda: FormulaParser('y~x', ["x#", "y"]))
        #test removing spaces
        model_formula = FormulaParser('y~ x ', ["y", "x"])
        self.assertEqual(model_formula.formula, 'y~x')


    def test_get_response(self):
        #check basic functionality
        model_formula = FormulaParser('y~x', ["y", "x"])
        self.assertEqual(model_formula.get_response(), 'y')
        #check if response is not in column names
        self.assertRaises(ValueError, lambda: FormulaParser('y~x', ["x"]))

    def test_get_predictors(self):
        test_cases = [
            (FormulaParser('y~x', ["x", "y"]).get_predictors(), ["Intercept","x"]),
            (FormulaParser('y_~x_1+x_2', ["x_1","x_2", "y_"]).get_predictors(), ["Intercept","x_1","x_2"]),
            (FormulaParser('y~x1+x2', ["x1","x2", "y"]).get_predictors(), ["Intercept","x1","x2"]),
            (FormulaParser('y~x2+x1', ["x1","x2", "y"]).get_predictors(), ["Intercept","x1","x2"]),
            (FormulaParser('y~.', ["x1", "x2", "y"]).get_predictors(), ["Intercept", "x1", "x2"]),
            (FormulaParser('y~.-x1', ["x1", "x2", "y"]).get_predictors(), ["Intercept", "x2"]),
            (FormulaParser('y~.+1', ["x1", "x2", "y"]).get_predictors(), ["Intercept", "x1", "x2"]),
            (FormulaParser('y~.-1', ["x1", "x2", "y"]).get_predictors(), ["x1", "x2"]),
            (FormulaParser('y~.+0', ["x1", "x2", "y"]).get_predictors(), ["x1", "x2"])
        ]
        for case, expect in test_cases:
            self.assertEqual(case,expect)


    def test_has_intercept(self):
        test_cases = [
            (FormulaParser('y~x', ["x", "y"]).has_intercept(), True),
            (FormulaParser('y~x1+x2', ["x1", "x2", "y"]).has_intercept(), True),
            (FormulaParser('y~x2+x1', ["x1", "x2", "y"]).has_intercept(), True),
            (FormulaParser('y~.', ["x1", "x2", "y"]).has_intercept(), True),
            (FormulaParser('y~.-x1', ["x1", "x2", "y"]).has_intercept(), True),
            (FormulaParser('y~.+1', ["x1", "x2", "y"]).has_intercept(), True),
            (FormulaParser('y~.-1', ["x1", "x2", "y"]).has_intercept(), False),
            (FormulaParser('y~.+0', ["x1", "x2", "y"]).has_intercept(), False)
        ]
        for case, expect in test_cases:
            self.assertEqual(case, expect)

df1 = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})

df2 = pd.DataFrame({"x": [1, 1, 1, 1], "y": [0, 1, 2, 3]})

df3 = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [4, 5, 6, 7, 8]})

class BaseModelTest(TestCase):
    def test_init(self):
        my_model = BaseModel("y~.", df1)
        self.assertRaises(NotImplementedError, lambda: my_model.fit())
        self.assertRaises(NotImplementedError, lambda: my_model.summary())
        self.assertRaises(NotImplementedError, lambda: my_model.plot())
        self.assertRaises(TypeError, lambda: my_model.predict())
        self.assertRaises(NotImplementedError, lambda: my_model.predict(df1))

    def test_getX(self):
        my_model = BaseModel("y~.", df1)
        test_cases = [
            (BaseModel("y~.", df1).getX(df1), pd.DataFrame({"x": [0, 0, 1, 1], "Intercept": [1, 1, 1, 1]})),
            (BaseModel("y~.-1", df1).getX(df1), pd.DataFrame({"x": [0, 0, 1, 1]}))
        ]
        for case, expect in test_cases:
            assert case.equals(expect)

class BaseRegressorTest(TestCase):
    def test_init(self):
        my_model = BaseRegressor("y~.",df1)
        pass

class BaseClassifierTest(TestCase):
    def test_init(self):
        my_model = BaseClassifier("y~.",df1)
        self.assertRaises(NotImplementedError, lambda:my_model.predict_proba(df1))

class LMTest(TestCase):
    def test_lm_fit(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        self.assertEqual(my_lm.intercept, 1.0)
        self.assertEqual(my_lm.coefs[0], 3.5)

        my_lm = lm("y~.-1", data=df2)
        my_lm.fit()
        self.assertEqual(my_lm.coefs[0], 1.5)

    def test_lm_predict(self):
        input1 = pd.DataFrame({"x": [0, 1]})
        expected1 = np.array([1,4.5])

    def test_lm_summary(self):

        pass

    def test_lm_plot(self):
        pass

    # def test_model_invalid_input_predictors(self):
    #     model_formula = FormulaParser('dist~sp@ed',  ["speed", "dist"])
    #     self.assertRaises(model_formula.formula, 'dist~speed')

        # df = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})
        # display(df)
        # my_lm = lm("y~.", data=df)
        # my_lm = lm("y~x",data = df)
        # no intercept models
        # my_lm = lm("y~.-1",data = df)
        # my_lm = lm("y~.+0",data = df)
        # my_lm.fit()
        # print("coefs", my_lm.coefs)
        # print("std errors:", my_lm.std_error)


class HelperTest(TestCase):

    def test_rounded_string(self):
        pass
