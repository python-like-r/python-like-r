from functools import wraps
from time import time


def timing(f):
    """
    Decorator to print the time taken to run the method
    :param f: function it is wrapped around
    :return: result the function it is wrapped around
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        start_time = time()
        result = f(*args, **kwargs)
        end_time = time()
        print('%s function took %0.3f ms' % (f.func_name, (end_time - start_time) * 1000.0))
        return result

    return wrap


def is_isalnum_or_in_str(c, s):
    """
    checks if a character is a-z, A-Z, 0-9, or in the string s.
    :return: True if c is alphanumaric or in s.
    """
    return c.isalnum() or c in s


def is_valid_colname(s):
    """
    checks that a string only contains alphanumeric chars and underscores.
    :return: True if all chars pass.
    """
    return all(map(lambda c: is_isalnum_or_in_str(c, "_"), s))


def rounded_str(num):
    """
    round the model values to 3 digits and return string

    :param self:
    :param num:
    :return: string rounded to 3 digits
    """
    return str(round(num, 3))


def get_p_significance(p_value):
    """
    Get the statistical significance indicator
    :param p_value:
    :return: statistical significance indicator
    """
    if 0 < p_value < 0.001:
        return '***'
    elif 0.001 < p_value < 0.01:
        return '**'
    elif 0.01 < p_value < 0.05:
        return '*'
    elif 0.05 < p_value < 0.1:
        return '.'
    elif 0.1 < p_value < 1:
        return ' '


