import pandas as pd

from unittest import TestCase
from src.utility.FormulaParser import FormulaParser


class FormulaTest(TestCase):

    def test_model_input_formula(self):
        model_formula = FormulaParser('dist~speed',  ["speed", "dist"])
        self.assertEqual(model_formula.formula, 'dist~speed')
        # self.assertCountEqual(len(model_formula.predictors), 2)
        self.assertEqual(model_formula.response, 'dist')

    # def test_model_invalid_input_formula(self):
    #     # model_formula = FormulaParser('dist.speed',  ["speed", "dist"])
    #     self.assertRaises(ValueError, FormulaParser('dist.speed',  ["speed", "dist"]))

    def test_model_input_formula_with_pred(self):
        model_formula = FormulaParser('y~x',  ["y", "x"])
        self.assertEqual(model_formula.formula, 'y~x')

    def test_model_input_formula_no_intersect(self):
        df = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 2, 4, 5]})
        model_formula = FormulaParser('y~.-1',  ["y", "x"])
        self.assertEqual(model_formula.predictors, ['x'])


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
