"""Test some general properties of all recipes."""
import pytest
from asr.core import get_recipes

all_recipes = get_recipes()


@pytest.mark.parametrize("recipe", all_recipes)
def test_recipe_cli_help_calls(asr_tmpdir, capsys, recipe):
    """Test that all help calls actually works."""
    recipe.cli(['-h'])
    captured = capsys.readouterr()
    name = recipe.name
    assert f'Usage: asr run {name}' in captured.out


@pytest.mark.parametrize("recipe", all_recipes, ids=lambda x: x.name)
def test_recipe_cli_types(asr_tmpdir, capsys, recipe):
    """Test that all parameters have been given types."""
    params = recipe.get_parameters()

    notypes = set()
    for param in params.values():
        if 'type' not in param:
            notypes.add(param['name'])

    assert not notypes
