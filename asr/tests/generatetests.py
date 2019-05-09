from pathlib import Path
from asr.utils import get_recipes

recipes = get_recipes()


for recipe in recipes:
    name = recipe.__name__.split('.')[1]
    template = (Path(__file__).parent / 'template.py').read_text()

    text = template.replace('###', name)

    testname = f'test_{name}.py'
    print(f'Writing {testname}')
    (Path(__file__).parent / testname).write_text(text)
