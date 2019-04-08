import click
from asr.utils import get_recipes


def summary(content, indent=0, title=None, pad=2):
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
            output += f'\n{row}\n'
            continue
        for colw, desc in zip(colwidth_c, row):
            out += f'{desc: <{colw}}' + ' ' * pad
        output += out
        output += '\n'

    return output


def check_recipes():
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

    pretty_output = summary(panel)
    print(pretty_output)


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
def list():
    """List all recipes"""
    check_recipes()
