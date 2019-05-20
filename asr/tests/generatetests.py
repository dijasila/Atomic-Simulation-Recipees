def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes

    recipes = get_recipes(group='property')

    for recipe in recipes:
        name = recipe.__name__.split('.')[1]
        if name == 'collect':
            continue
        template = (Path(__file__).parent / 'template.py').read_text()

        text = template.replace('###', name)

        testname = f'test_auto_{name}.py'
        print(f'Writing {testname}')
        (Path(__file__).parent / testname).write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_auto_*.py')

    for p in paths:
        p.unlink()
