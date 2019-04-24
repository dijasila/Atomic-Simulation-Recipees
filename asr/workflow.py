from pathlib import Path
import click


@click.command()
def main():
    """Run all recipes"""
    from asr import quickinfo
    from ase.parallel import parprint, world
    parprint('Quickinfo...')
    quickinfo.main(standalone_mode=False)

    from asr import gs
    parprint('Ground state...')
    gs.main(standalone_mode=False)

    from asr import convex_hull
    parprint('Convex hull...')
    references = str(Path('~/oqmd12.db').expanduser())
    print(references)
    convex_hull.main(standalone_mode=False,
                     args=['--references', references])

    from asr import gaps
    parprint('Band gaps...')
    gaps.main(standalone_mode=False)
  
    from asr import bandstructure 
    parprint('Band structure...')
    bandstructure.main(standalone_mode=False)

    from asr import bader
    parprint('Bader analysis...')
    bader.main(standalone_mode=False)

    from asr import dos
    parprint('Density of states...')
    dos.main(standalone_mode=False)

    from asr import collect
    parprint('Collect data...')
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
    main(standalone_mode=False)
