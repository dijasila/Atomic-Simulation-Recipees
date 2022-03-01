"""Test some general properties of all recipes."""
import typing
import pytest
from ase import Atoms
from asr.core import get_recipes, DictStr, AtomsFile
import click

pytestmark = pytest.mark.xfail('not now')

all_recipes = []  # XXX get_recipes()


@pytest.mark.ci
@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
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
    params = recipe.get_argument_descriptors()

    notypes = set()
    for param in params.values():
        if 'type' not in param and 'is_flag' not in param:
            notypes.add(param['name'])
    assert not notypes


@pytest.mark.ci
@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
def test_recipe_type_hints(asr_tmpdir, capsys, recipe):
    """Test that all parameters have been given types."""
    params = recipe.get_argument_descriptors()

    notypes = set()
    for param in params.values():
        if 'type' not in param and 'is_flag' not in param:
            notypes.add(param['name'])
    assert not notypes

    func = recipe.get_wrapped_function()
    type_hints = typing.get_type_hints(func)

    type_hints_that_should_exist = set(params)
    type_hints_that_should_exist.add('return')

    assert set(type_hints) == type_hints_that_should_exist, \
        f'Missing type hints: {recipe.name}'
