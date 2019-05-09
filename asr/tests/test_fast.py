from asr.utils import get_recipes
import pytest
from pathlib import Path

folder = Path(__file__).parent

# Can we get all recipes?
recipes = get_recipes()


@pytest.mark.parametrize('recipe', recipes)
def test_help(self, recipe):
    """Call all main functions with --help"""
    if not hasattr(recipe, 'main'):
        return

    try:
        func = recipe.main
        func(args=['--help'])
    except Exception:
        print(f'Problem in function {recipe.__name__}.main '
              'when called with --help')
        raise


@pytest.mark.parametrize('recipe', recipes)
def test_asr_command(self, recipe):
    """Make sure that the correct _asr_command is being used"""
    if hasattr(recipe, 'main'):
        try:
            assert hasattr(recipe.main, '_asr_command')
        except AssertionError:
            msg = ('Dont use @click.command! Please use '
                   'the "from asr.utils import command" '
                   'in stead')
            raise AssertionError(msg)


@pytest.mark.parametrize('recipe', recipes)
def test_collect(self, recipe):
    """Call all collect_data functions with empty list
    (should work)"""

    if not hasattr(recipe, 'collect_data'):
        return

    try:
        recipe.collect_data([])
    except Exception:
        print(f'Problem in function {recipe.__name__}.collect_data')
        raise
