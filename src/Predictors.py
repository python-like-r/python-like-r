from src.validate_descriptor import ValidColumnNames


def predictors(attr):
    def decorator(cls):
        setattr(cls, attr, ValidColumnNames)
        return cls
    return decorator


@predictors("column_names")
class Predictors:
    def __init__(self, column_names):
        self.predictors = column_names
