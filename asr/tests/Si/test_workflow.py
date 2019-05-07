from asr.utils.recipe import Recipe
from asr.utils import get_recipes
from pathlib import Path
import pytest


recipes = get_recipes(sort=True)


@pytest.fixture(scope='class')
def cleanfolder():
    import shutil
    for p in Path('.').glob('*'):
        if (p.name.startswith('test_') or
            p.name in ['start.json', 'params.json']):
            continue
        if p.is_dir():
            shutil.rmtree(p.name)
            continue
        p.unlink()
    yield


class TestWorkflow():
    ...


for recipe in recipes:
    if recipe.__name__ == 'asr.workflow':
        continue

    name = recipe.__name__

    def func(cls, cleanfolder, name=name):
        recipe = Recipe.frompath(name)
        recipe.main(args=[])

    setattr(TestWorkflow, f'test_{name}', func)
