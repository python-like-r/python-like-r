from unittest import TestCase
from src.Formula import Formula
from src.Predictors import Predictors


class FormulaTest(TestCase):

    def test_model_input_formula(self):
        model_formula = Formula('dist~speed')
        self.assertEqual(model_formula.formula, 'dist~speed')

    def test_model_input_column_names(self):
        model_predictors = Predictors(['spped', 'mileage'])
        self.assertEqual(model_predictors.predictors, ['spped', 'mileage'])
