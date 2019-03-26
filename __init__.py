import importlib
from pathlib import Path

files = Path(__file__).parent.glob('*.py')

exclude = ['__init__.py', '__main__.py']

recipes = []
for file in files:
    is_recipe = True
    if str(file.name) in exclude:
        is_recipe = False

    if is_recipe:
        name = file.with_suffix('').name
        module = importlib.import_module(f'asr.{name}')
        recipes.append(module)


