import pandas as pd

from unittest import TestCase
from src.models.BaseModel import BaseModel
from src.models.BaseRegressor import BaseRegressor
from src.models.BaseClassifier import BaseClassifier
from src.models.lm import lm
from src.utility.FormulaParser import FormulaParser
from src.utility.helper import rounded_str, get_p_significance


class FormulaParserTest(TestCase):

    def test_init(self):
        # test invalid chars
        self.assertRaises(ValueError, lambda: FormulaParser('y~x%', ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('y~x!', ["x", "y"]))
        # test number of "~" is not 1
        self.assertRaises(ValueError, lambda: FormulaParser('y~~x',  ["x", "y"]))
        self.assertRaises(ValueError, lambda: FormulaParser('yx', ["x", "y"]))
        # test list properties
        self.assertRaises(ValueError, lambda: FormulaParser('y~x', ["x#", "y"]))
        # test removing spaces
        model_formula = FormulaParser('y~ x ', ["y", "x"])
        self.assertEqual(model_formula.formula, 'y~x')

    def test_get_response(self):
        # check basic functionality
        model_formula = FormulaParser('y~x', ["y", "x"])
        self.assertEqual(model_formula.get_response(), 'y')
        # check if response is not in column names
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
        predict_df = pd.DataFrame({"x": [0, 1]})
        expected_y = [1, 4.5]

        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        y_hat = my_lm.predict(predict_df, interval=None)
        self.assertListEqual(y_hat.tolist(), expected_y)

    def test_lm_predict_c(self):
        predict_df = pd.DataFrame({"x": [0, 1]})
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        self.assertRaises(NotImplementedError, lambda: my_lm.predict(predict_df, interval='c'))

    def test_lm_predict_p(self):
        predict_df = pd.DataFrame({"x": [0, 1]})
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        self.assertRaises(NotImplementedError, lambda: my_lm.predict(predict_df, interval='p'))

    def test_lm_summary(self):
        formula = "y~."
        my_lm = lm(formula, data=df1)
        my_lm.fit()
        summary = my_lm.summary(print_summary=False)
        lm_summary_formula = summary.splitlines()[2].split(',')[0].split('=')[1].strip()
        self.assertEqual(formula, lm_summary_formula)

    def test_lm_plot_1(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        my_lm.plot(which=1)
        pass

    def test_lm_plot_2(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        my_lm.plot(which=2)
        pass

    def test_lm_plot_3(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        my_lm.plot(which=3)
        pass

    def test_lm_plot_4(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        self.assertRaises(NotImplementedError, lambda: my_lm.plot(which=4))

    def test_lm_plot_all(self):
        my_lm = lm("y~.", data=df1)
        my_lm.fit()
        self.assertRaises(NotImplementedError, lambda: my_lm.plot(which=None))


class HelperTest(TestCase):

    def test_rounded_string(self):
        self.assertEqual(rounded_str(0.0036), '0.004')

    def test_p_significance_3_star(self):
        self.assertEqual(get_p_significance(0.0009), ' ***')

    def test_p_significance_2_star(self):
        self.assertEqual(get_p_significance(0.009), ' **')

    def test_p_significance_1_star(self):
        self.assertEqual(get_p_significance(0.09), ' .')

    def test_p_significance_0_star(self):
        self.assertEqual(get_p_significance(0.9), ' ')
