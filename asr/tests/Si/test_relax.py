from asr.utils.recipe import Recipe
from pathlib import Path

recipe = Recipe.frompath('asr.relax')


def cleanfolder():
    import shutil
    for p in Path('.').glob('*'):
        if p.name in ['start.json', 'params.json']:
            continue
        if p.is_dir():
            shutil.rmtree()
        p.unlink()


def test_no_params():
    """Test gs with no params"""
    cleanfolder()
    func = recipe.main
    func(args=[])


# def test_gs():
#     cleanfolder()
#     func = recipe.main
#     func(args=[])
