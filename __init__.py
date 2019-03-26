import importlib
from pathlib import Path

files = Path(__file__).parent.glob('*.py')

recipes = []
for file in files:
    is_recipe = False
    with open(str(file), 'r') as fd:
        for line in fd:
            if line.startswith('def main('):
                is_recipe = True
                break
    if is_recipe:
        name = file.with_suffix('').name
        module = importlib.import_module(f'asr.{name}')
        recipes.append(module)


