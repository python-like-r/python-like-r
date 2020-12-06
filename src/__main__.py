from src.validate_descriptor import ModelInputs

if __name__ == "__main__":
    model_inputs = ModelInputs('dist~speed', ['mileage'])
    print(model_inputs.formula)
    print(model_inputs.columns)
