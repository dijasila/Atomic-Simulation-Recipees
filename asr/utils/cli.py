import click
from asr.utils import get_recipes


def format(content, indent=0, title=None, pad=2):
    colwidth_c = []
    for row in content:
        if isinstance(row, str):
            continue
        for c, element in enumerate(row):
            nchar = len(element)
            try:
                colwidth_c[c] = max(colwidth_c[c], nchar)
            except IndexError:
                colwidth_c.append(nchar)

    output = ''
    if title:
        output = f'\n{title}\n'
    for row in content:
        out = ' ' * indent
        if isinstance(row, str):
            output += f'\n{row}'
            continue
        for colw, desc in zip(colwidth_c, row):
            out += f'{desc: <{colw}}' + ' ' * pad
        output += out
        output += '\n'

    return output


@click.group()
def cli():
    ...


@cli.command()
@click.option('--database', default='database.db')
@click.option('--custom', default='asr.utils.custom')
def browser(database, custom):
    """Open results in web browser"""
    import subprocess
    from pathlib import Path

    if custom == 'asr.utils.custom':
        custom = Path(__file__).parent / 'custom.py'

    cmd = f'python3 -m ase db {database} -w -M {custom}'
    print(cmd)
    subprocess.run(cmd.split())


@cli.command()
@click.option('--collect', default=False, is_flag=True,
              help='Only collect tests')
def test(collect):
    """Run test of recipes"""
    from pathlib import Path
    import subprocess
    import asr
    folder = str(Path(asr.__file__).parent)

    cmd = f'python3 -m pytest --tb=short {folder}'
    if collect:
        cmd += ' --collect-only'
    print(cmd)
    subprocess.run(cmd.split())


@cli.command()
def status():
    """Show status of current directory"""
    from pathlib import Path
    recipes = get_recipes()
    panel = []
    missing_files = []
    for recipe in recipes:
        status = [recipe.__name__]
        done = True
        if hasattr(recipe, 'creates'):
            for create in recipe.creates:
                if not Path(create).exists():
                    done = False
            if done:
                status.append(f'Done -> {recipe.creates}')
            else:
                status.append(f'Todo')
            if done:
                panel.insert(0, status)
            else:
                panel.append(status)
        else:
            status.append('No files created')
            missing_files.append(status)
    
    print(format(panel))
    print(format(missing_files))


@cli.command()
@click.argument('name')
def check(name):
    """Get a detailed description of a recipe"""
    from asr.utils.recipe import Recipe
    recipe = Recipe.frompath(name)
    print(recipe)


@cli.command()
def checkall():
    """Check status of all recipes"""
    recipes = get_recipes()

    attributes = ['main',
                  'creates',
                  'collect_data',
                  'webpanel',
                  'resources']

    groups = ['Structure', 'Property',
              'Postprocessing', 'Utility']
    panel = []
    panel.append(['name', *attributes])
    for group in groups:
        panel.append(f'{group} recipes')
        for recipe in recipes:
            if not recipe.group == group:
                continue
            status = [recipe.__name__]
            for attr in attributes:
                if hasattr(recipe, attr):
                    status.append('.')
                else:
                    status.append('N')
            panel.append(status)

    pretty_output = format(panel)
    print(pretty_output)


@cli.command()
@click.argument('recipe')
def plot(recipe):
    """Plot figures interactively"""
    import importlib
    from ase.db import connect
    from matplotlib import pyplot as plt
    
    module = importlib.import_module(recipe)
    db = connect('database.db')

    rows = list(db.select())
    for row in rows:
        _, things = module.webpanel(rows[-1], {})

        for func, names in things:
            func(row, *names)

    plt.show()
