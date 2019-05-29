# Test a full workflow

from asr.utils.recipe import Recipe
from asr.utils import get_dep_tree, chdir
from pathlib import Path
import pytest

# Test name
recipename = '###'

# We create temporary directory and move the structure.json
# and params.json into that directory
@pytest.fixture(scope='class')
def directory(tmpdir_factory):
    path = tmpdir_factory.mktemp(f'test_{recipename}')
    srcstart = Path(__file__).parent / 'Si.json'
    srcparams = Path(__file__).parent / 'small_params.json'
    dststart = Path(path) / 'unrelaxed.json'
    dstparams = Path(path) / 'params.json'
    dststart.write_bytes(srcstart.read_bytes())
    dstparams.write_bytes(srcparams.read_bytes())
    return path


class TestWorkflow():
    ...


def relax(cls, directory):
    with chdir(directory):
        # Make sure to reload the module
        recipe = Recipe.frompath('asr.relax', reload=True)
        recipe.main(args=[])


setattr(TestWorkflow, f'test_relax', relax)

# Dynamically add tests of all recipes
recipes = get_dep_tree(f'asr.{recipename}')
for recipe in recipes:
    name = recipe.__name__

    if recipe.name == 'asr.relax':
        continue

    if recipe.group:
        if recipe.group == 'postprocessing':
            continue

    def func(cls, directory, name=name):
        with chdir(directory):
            # Make sure to reload the module
            recipe = Recipe.frompath(name, reload=True)
            recipe.main(args=[])

    setattr(TestWorkflow, f'test_{name}', func)


def collect(cls, directory):
    with chdir(directory):
        # Make sure to reload the module
        recipe = Recipe.frompath('asr.collect', reload=True)
        recipe.main(args=[])


def browser(cls, directory):
    with chdir(directory):
        # Make sure to reload the module
        from asr.utils.cli import browser
        browser(standalone_mode=False, args=['--only-figures'])


# Finally collect
setattr(TestWorkflow, f'test_collect', collect)
setattr(TestWorkflow, f'test_browser', browser)
