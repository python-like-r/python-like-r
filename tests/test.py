from unittest import TestCase
from src.Formula import Formula


class FormulaTest(TestCase):

    def test_formula(self):
        formula = Formula('dist~speed', ['mileage'])
        self.assertEqual(formula.formula, 'dist~speed')

    def test_column_names(self):
        formula = Formula('dist~speed', ['mileage'])
        self.assertEqual(formula.columns, ['mileage'])
