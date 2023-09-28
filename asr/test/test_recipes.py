"""Test some general properties of all recipes."""
import typing
import pytest
from ase import Atoms
from asr.core import get_recipes, DictStr, AtomsFile
import click

all_recipes = get_recipes()


@pytest.mark.ci
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


@pytest.mark.ci
@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
def test_recipe_use_new_webpanel_implementation(recipe):
    assert recipe.webpanel is None
