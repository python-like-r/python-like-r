from unittest import TestCase
from src.validate_descriptor import ModelInputs


class FormulaTest(TestCase):

    def test_model_input_formula(self):
        model_inputs = ModelInputs('dist~speed', ['mileage'])
        self.assertEqual(model_inputs.formula, 'dist~speed')

    def test_model_input_column_names(self):
        model_inputs = ModelInputs('dist~speed', ['mileage'])
        self.assertEqual(model_inputs.columns, ['mileage'])
