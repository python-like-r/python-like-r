from unittest import TestCase
from src.FormulaParser import FormulaParser


class FormulaTest(TestCase):

    def test_model_input_formula(self):
        model_formula = FormulaParser('dist~speed',  ["speed", "dist"])
        self.assertEqual(model_formula.formula, 'dist~speed')
        self.assertEqual(model_formula.predictors, ['speed'])
        self.assertEqual(model_formula.response, 'dist')

    def test_model_invalid_input_formula(self):
        model_formula = FormulaParser('dist.speed',  ["speed", "dist"])
        self.assertRaises(model_formula.formula, 'dist~speed')

    def test_model_invalid_input_predictors(self):
        model_formula = FormulaParser('dist~sp@ed',  ["speed", "dist"])
        self.assertRaises(model_formula.formula, 'dist~speed')