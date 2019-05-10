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


@cli.command(context_settings={'ignore_unknown_options': True,
                               'allow_extra_args': True})
@click.pass_context
def test(ctx):
    """Run test of recipes"""
    import subprocess
    from asr.tests.generatetests import generatetests
    generatetests()
    args = ctx.args

    if not args:
        args = ['asr.tests']
    cmd = f'python3 -m pytest --pyargs asr ' + ' '.join(args)
    print(cmd)
    subprocess.run(cmd.split())


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
