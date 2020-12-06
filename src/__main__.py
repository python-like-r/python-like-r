from src.Formula import Formula
from src.Predictors import Predictors

if __name__ == "__main__":
    model_formula = Formula('dist~speed')
    model_predictors = Predictors(['spped', 'mileage'])
    print(model_formula.formula)
    print(model_predictors.predictors)
