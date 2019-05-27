from asr.utils import get_recipes
import pytest
from pathlib import Path

folder = Path(__file__).parent

# Can we get all recipes?
recipes = get_recipes()


@pytest.mark.parametrize('recipe', recipes)
def test_help(recipe):
    """Call all main functions with --help"""

    if not recipe.main:
        return

    try:
        func = recipe.main
        func(args=['--help'])
    except Exception:
        print(f'Problem in function {recipe.__name__}.main '
              'when called with --help')
        raise


@pytest.mark.parametrize('recipe', recipes)
def test_group(recipe):
    """Make sure that the group property is implemented"""
    if not recipe.group:
        return

    assert recipe.group in ['structure', 'property', 'postprocessing',
                            'setup'], \
        (f'Group {recipe.__name__} not known!')


@pytest.mark.parametrize('recipe', recipes)
def test_asr_command(recipe):
    """Make sure that the correct _asr_command is being used"""
    if not recipe.main:
        return
    
    try:
        assert hasattr(recipe.main, '_asr_command')
    except AssertionError:
        msg = ('Dont use @click.command! Please use '
               'the "from asr.utils import command" '
               'in stead')
        raise AssertionError(msg)
