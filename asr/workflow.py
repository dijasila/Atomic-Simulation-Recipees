from pathlib import Path
import click


@click.command()
def main():
    """Run all recipes"""
    from asr import quickinfo
    print('Quickinfo...')
    quickinfo.main(standalone_mode=False)

    from asr import gs
    print('Ground state...')
    gs.main(standalone_mode=False)

    from asr import convex_hull
    print('Convex hull...')
    convex_hull.main(standalone_mode=False, args=['--references', 'oqmd12.db'])

    from asr import gaps
    print('Band gaps...')
    gaps.main(standalone_mode=False)
  
    from asr import bandstructure 
    print('Band structure...')
    bandstructure.main(standalone_mode=False)

    from asr import bader
    print('Bader analysis...')
    bader.main(standalone_mode=False)

    from asr import dos
    print('Density of states...')
    dos.main(standalone_mode=False)

    from asr import collect
    print('Collect data...')
    collect.main(standalone_mode=False, args=['.'])


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
