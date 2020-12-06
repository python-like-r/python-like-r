from src.FormulaParser import FormulaParser

if __name__ == "__main__":
    model_formula = FormulaParser('dist~speed', ["speed", "dist"])
    print('lm formula in main: ', model_formula.formula)
    print('lm predictors in main: ', model_formula.predictors)
    print('lm response in main: ', model_formula.response)
