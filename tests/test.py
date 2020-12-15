import pandas as pd

from unittest import TestCase
from src.utility.FormulaParser import FormulaParser


class FormulaTest(TestCase):

    def test_input_formula(self):
        #test invalid chars
        self.assertRaises(ValueError, lambda: FormulaParser('y~x%', ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('y~x!', ["x", "y"]))
        #test number of "~" is not 1
        self.assertRaises(ValueError,lambda: FormulaParser('y~~x',  ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('yx', ["x", "y"]))
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


class LMTest(TestCase):

    def test_lm_fit(self):
        pass

    def test_lm_predict(self):
        pass

    def test_lm_summary(self):
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
