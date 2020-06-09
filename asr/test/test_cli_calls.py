"""Test some general properties of all recipes."""
import typing
import pytest
from asr.core import get_recipes
import click

all_recipes = get_recipes()


@pytest.mark.parametrize("recipe", all_recipes)
def test_recipe_cli_help_calls(asr_tmpdir, capsys, recipe):
    """Test that all help calls actually works."""
    recipe.cli(['-h'])
    captured = capsys.readouterr()
    name = recipe.name
    assert f'Usage: asr run {name}' in captured.out


map_types = {typing.List[str]: str,
             typing.List[float]: float,
             typing.Union[str, None]: str}


@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
def test_recipe_cli_types(asr_tmpdir, capsys, recipe):
    """Test that all parameters have been given types."""
    params = recipe.get_parameters()

    notypes = set()
    for param in params.values():
        if 'type' not in param and 'is_flag' not in param:
            notypes.add(param['name'])
    assert not notypes


@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
def test_recipe_type_hints(asr_tmpdir, capsys, recipe):
    """Test that all parameters have been given types."""
    params = recipe.get_parameters()

    notypes = set()
    for param in params.values():
        if 'type' not in param and 'is_flag' not in param:
            notypes.add(param['name'])
    assert not notypes

    func = recipe.get_wrapped_function()
    type_hints = typing.get_type_hints(func)
    assert set(type_hints) == set(params), f'Missing type hints: {recipe.name}'
    # for name, param in params.items():
    #     if 'is_flag' in param:
    #         tp = bool
    #     else:
    #         tp = param['type']
    #     tp2 = type_hints[name]
    #     if tp2 in map_types:
    #         tp2 = map_types[tp2]
    #     assert tp == tp2, name
