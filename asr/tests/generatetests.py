def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes, get_dep_tree

    recipes = get_recipes(group='property')

    for material in ['Si.json', 'Fe.json',
                     'BN.json', 'VS2.json']:
        for recipe in recipes:
            name = recipe.name.split('.')[1]
            text = (Path(__file__).parent / 'template.py').read_text()
            text = text.replace('# Material #', material)
            depsection = ''
            for dep in get_dep_tree(f'{recipe.name}'):
                line = f"Recipe.frompath('{dep.name}', reload=True).run()\n"
                depsection += line

            formula = material.split('.')[0]
            text = text.replace('# DepSection #', depsection)
            testname = f'test_auto_{formula}_{name}.py'
            print(f'Writing {testname}')
            (Path(__file__).parent / testname).write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_auto_*.py')

    for p in paths:
        p.unlink()
