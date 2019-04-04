from pathlib import Path
import click


@click.command()
def main():
    """Run all recipes"""
    pass


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
