from asr.utils import get_recipes
import pytest
from pathlib import Path

folder = Path(__file__).parent

# Can we get all recipes?
recipes = get_recipes()


class TestImports():
    @pytest.mark.parametrize('recipe', recipes)
    def test_imports(self, recipe):
        """Call all main functions with --help"""
        print(recipe.__name__)
        # Make sure that the correct
        if hasattr(recipe, 'main'):
            try:
                assert hasattr(recipe.main, '_asr_command')
            except AssertionError:
                msg = ('Dont use @click.command! '
                       'Please use the "from asr.utils import command"'
                       ' in stead')
                raise AssertionError(msg)
