def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes, get_dep_tree

    recipes = get_recipes(group='property')

    for recipe in recipes:
        name = recipe.__name__.split('.')[1]
        template = (Path(__file__).parent / 'template.py').read_text()

        text = template.replace('###', name)

        depsection = ''
        for dep in get_dep_tree(f'{recipe.name}'):
            depsection += f"Recipe.frompath('{dep.name}', reload=True).run()\n"

        text = text.replace('# DepSection #', depsection)
        testname = f'test_auto_{name}.py'
        print(f'Writing {testname}')
        (Path(__file__).parent / testname).write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_auto_*.py')

    for p in paths:
        p.unlink()
