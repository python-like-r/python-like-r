from src.helper import is_isalnum_or_in_str


class ValidFormula:

    def __get__(self, obj):
        self.value

    def __set__(self, obj, value):
        if not all(map(lambda c: is_isalnum_or_in_str(c, "_()~.+-"), value)):
            raise ValueError("Invalid character found in formula.")
        if value.count("~") != 1:
            raise ValueError("Formula should contain exactly one '~'.")
        self.value = value


class ValidColumnNames:

    @staticmethod
    def is_valid_colname(s):
        """
        checks that a string only contains alphanumeric chars and underscores.
        :return: True if all chars pass.
        """
        return all(map(lambda c: is_isalnum_or_in_str(c, "_"), s))

    def __get__(self, obj):
        self.value

    def __set__(self, obj, value):
        if not all(map(value.is_valid_colname)):
            raise ValueError("Expected all column names to only contain alphanum and underscores.")
        self.value = value
