from src.helper import is_isalnum_or_in_str, is_valid_colname


class ValidFormula:
    def __init__(self):
        self.__formula = ''

    def __get__(self, instance, owner):
        return self.__formula

    def __set__(self, instance, value):
        if not all(map(lambda c: is_isalnum_or_in_str(c, "_()~.+-"), value)):
            raise ValueError("Invalid character found in formula.")
        if value.count("~") != 1:
            raise ValueError("Formula should contain exactly one '~'.")
        self.__formula = value


class ValidColumnNames:
    def __init__(self):
        self.__columns = []

    def __get__(self, instance, owner):
        return self.__columns

    def __set__(self, instance, value):
        if not all(map(is_valid_colname, value)):
            raise ValueError("Expected all column names to only contain alphanum and underscores.")
        self.__columns = value
