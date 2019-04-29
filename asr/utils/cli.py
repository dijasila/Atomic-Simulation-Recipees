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
def test():
    """Run test of recipes"""
    from pathlib import Path
    from subprocess import Popen, PIPE
    import asr
    folder = str(Path(asr.__file__).parent)

    with Popen(['python3', '-m', 'pytest',
                '--tb=short',  # shorter traceback format
                folder], stdout=PIPE, bufsize=1,
               universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')


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
    import inspect
    lines = inspect.getsource(recipe.main.callback)
    print(lines)
    import difflib
    # template = Recipe.frompath('asr.utils.template')
    # tmplines = inspect.getsource(template.main.callback)

    # s = difflib.SequenceMatcher(isjunk=lambda x: x in ['*', 'pass'],
    #                             a=lines,
    #                             b=tmplines)
    # for block in s.get_matching_blocks():
    #     i, j, n = block
    #     print('a')
    #     print(lines[i:i + n])
    #     print('b')
    #     print(tmplines[j:j + n])
    #     # print("a[%d] and b[%d] match for %d elements" % block)


@cli.command()
def printdependencies():
    pass


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
