from pathlib import Path
from asr.utils import command


@command('asr.workflow')
def main():
    """Run all recipes"""
    from ase.parallel import parprint
    from asr.utils import get_recipes
    recipes = get_recipes(sort=True)
    for recipe in recipes:
        if 'asr.workflow' == recipe.__name__:
            continue

        # Check if some files are missing
        if hasattr(recipe, 'creates'):
            exists = [Path(create).exists() for create in recipe.creates]
            if all(exists):
                continue

        if recipe.group not in ['Structure', 'Property']:
            continue

        parprint(f'{recipe.__name__}...')
        try:
            recipe.main()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as x:
            print(x)
        parprint()


folder = Path(__file__).parent.resolve()
files = [str(path) for path in folder.glob('[a-zA-Z]*.py')]
recipes = []
for file in files:
    if 'workflow' in file:
        continue
    is_recipe = False
    with open(file, 'r') as fd:
        for line in fd:
            if line.startswith('resources = '):
                is_recipe = True
                break
    if is_recipe:
        name = Path(file).with_suffix('').name
        recipes.append(f'asr.{name}')

group = 'Structure'
dependencies = recipes
resources = '1:1m'

if __name__ == '__main__':
    main()
