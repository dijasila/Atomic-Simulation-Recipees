"""Test some general properties of all recipes."""
import typing
import pytest
from ase import Atoms
from asr.core import get_recipes, DictStr, AtomsFile
import click

all_recipes = get_recipes()


@pytest.mark.parametrize("recipe", all_recipes)
def test_recipe_cli_help_calls(asr_tmpdir, capsys, recipe):
    """Test that all help calls actually works."""
    recipe.cli(['-h'])
    captured = capsys.readouterr()
    name = recipe.name
    assert f'Usage: asr run {name}' in captured.out


map_typing_types = {typing.Union[float, None]: float,
                    typing.Union[Atoms, None]: Atoms,
                    typing.List[str]: str,
                    typing.List[int]: int,
                    typing.List[float]: float,
                    typing.Union[str, None]: str}

map_click_types = {AtomsFile: lambda x: Atoms,
                   DictStr: lambda x: dict,
                   click.Choice: lambda x: type(x.choices[0]),
                   click.Tuple: lambda x: int,  # This is not true in general.
                   bool: bool,
                   int: int,
                   str: str,
                   float: float}


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

    for name, param in params.items():
        if 'is_flag' in param:
            tp = bool
        else:
            tp = param['type']

        # Cast special click types to primitive types
        if tp not in [str, int, float, bool]:
            tp = map_click_types[type(tp)](tp)

        tp2 = type_hints[name]

        if tp2 in map_typing_types:
            # Cast special typing types to primitive types
            tp2 = map_typing_types[tp2]
        assert tp == tp2, name
