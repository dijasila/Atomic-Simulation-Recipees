import pytest


def recipe():
    pass


def main():
    pass


@pytest.mark.parametrize("function,expected",
                         [(recipe, 'asr.test.test_core@recipe'),
                          (main, 'asr.test.test_core')])
def test_recipe_name_from_function(function, expected):
    from asr.core.core import get_recipe_name_from_function
    name = get_recipe_name_from_function(function)
    assert name == expected
