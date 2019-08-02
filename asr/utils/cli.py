import click
from asr.utils import get_recipes
from asr.utils import argument, option


stdlist = list


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


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    ...


@cli.command(context_settings={'ignore_unknown_options': True,
                               'allow_extra_args': True})
@click.argument('args', metavar=('[shell] [dry] command '
                                 '[ARGS] in [FOLDER] ...'),
                nargs=-1)
@click.option('-p', '--parallel', type=int, help='Run on NCORES')
@click.pass_context
def run(ctx, args, parallel):
    """Run recipe or shell command in multiple folders.

    Can run an ASR recipe or a shell command. For example, the syntax
    "asr run recipe" will run the relax recipe in the current folder.

    To run a shell script use the syntax "asr run shell echo Hello!".
    This example would run "echo Hello!" in the current folder.

    Provide extra arguments to the recipe using "asr run recipe --arg1
    --arg2".

    Run command in multiple using "asr run recipe in folder1/ folder2/".
    This is also compatible with input arguments to the current command
    through "asr run recipe --arg1 in folder1/ folder2/". Here the
    special keyword "in" serves as the divider between arguments and
    folders.

    If you dont actually wan't to run the command, i.e., if it is a
    dangerous command, then use the "asr run dry ..." syntax where ...
    could be any of the above commands. For example,
    "asr run dry shell echo Hello! in */" would run "echo Hello!" in all
    folders of the current directory.

    Examples:

    \b
    Run the relax recipe:
        asr run relax
    Specify an argument:
        asr run relax --ecut 600
    Run relax recipe in two folders sequentially:
        asr run relax in folder1/ folder2/
    Run a shell command in this folder:
        asr run shell ase convert gs.gpw structure.json
    Run a shell command in "folder1/":
        asr run shell ase convert gs.gpw structure.json in folder1/
    Don't actually do anything just show what would be done
        asr run dry shell mv str1.json str2.json in folder1/ folder2/

    """
    import subprocess
    from pathlib import Path

    shell = False
    dryrun = False
    # Consume known commands (limit to 10 tries)
    for i, arg in enumerate(args):
        if arg == 'shell':
            shell = True
        elif arg == 'dry':
            dryrun = True
        else:
            break
    args = args[i:]

    # Are there any folders?
    folders = None
    if 'in' in args:
        ind = args.index('in')
        folders = args[ind + 1:]
        args = args[:ind]

    python = 'python3'
    if parallel:
        assert not shell, \
            ('You cannot execute a shell command in parallel. '
             'Only supported for python modules.')
        python = f'mpiexec -np {parallel} gpaw-python'

    # Identify function that should be executed
    if shell:
        command = ' '.join(args)  # The arguments are actually the command
    else:
        # If not shell then we assume that the command is a call
        # to a recipe
        recipe, *args = args
        if ':' in recipe:
            recipe, function = recipe.split(':')
            command = (f'{python} -c "from asr.{recipe} import {function}; '
                       f'{function}()" ') + ' '.join(args)
        else:
            command = f'{python} -m asr.{recipe} ' + ' '.join(args)

    if folders:
        from asr.utils import chdir

        for folder in folders:
            with chdir(Path(folder)):
                if dryrun:
                    print(f'Would run "{command}" in {folder}')
                else:
                    print(f'Running {command} in {folder}')
                    subprocess.run(command, shell=True)
    else:
        if dryrun:
            print(f'Would run "{command}"')
        else:
            print(f'Running command: {command}')
            subprocess.run(command, shell=True, check=True)
            # We only raise errors when check=True

    if dryrun and folders:
        nfolders = len(folders)
        print(f'Total number of folder: {nfolders}')


@cli.command()
@click.argument('recipe', type=str)
def help(recipe):
    """See help for recipe"""
    from asr.utils.recipe import Recipe
    command = f'asr.{recipe}'
    recipename = recipe
    recipe = Recipe.frompath(command, reload=True)

    with click.Context(recipe.main, info_name=f'asr run {recipename}') as ctx:
        print(recipe.main.get_help(ctx))


@cli.command()
@click.argument('search', required=False)
def list(search):
    """Show a list of all recipes"""
    recipes = get_recipes(sort=True)
    panel = [['Recipe', 'Description'],
             ['------', '-----------']]

    for recipe in recipes:
        if recipe.main:
            with click.Context(recipe.main,
                               info_name=f'asr run {recipe.name}') as ctx:
                longhelp = recipe.main.get_help(ctx)
                shorthelp = recipe.main.get_short_help_str()
        else:
            longhelp = ''
            shorthelp = ''

        if search and search not in longhelp:
            continue
        status = [recipe.name[4:], shorthelp]
        panel += [status]
    print(format(panel))


@cli.command()
def status():
    """Show the status of the current folder for all ASR recipes"""
    from pathlib import Path
    recipes = get_recipes()
    panel = []
    missing_files = []
    for recipe in recipes:
        status = [recipe.name]
        done = True
        if recipe.creates:
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
    
    print(format(panel))
    print(format(missing_files))


@cli.command(context_settings={'ignore_unknown_options': True,
                               'allow_extra_args': True})
@argument('tests', nargs=-1, required=False)
@option('-P', '--parallel',
        metavar='NCORES',
        type=int,
        help='Run tests in parallel on NCORES')
@option('-k', '--patterns', type=str, metavar='PATTERN,PATTERN,...',
        help='Select tests containing PATTERN.')
@option('-j', '--jobs', type=int, metavar='JOBS', default=1,
        help='Run JOBS threads.  Each test will be executed '
        'in serial by one thread.  This option cannot be used '
        'for parallelization together with MPI.')
@option('-s', '--show-output', is_flag=True,
        help='Show standard output from tests.')
def test(tests, parallel, patterns, jobs, show_output):
    from asr.utils.testrunner import ASRTestRunner
    import os
    import sys
    from pathlib import Path
    from asr.tests.generatetests import generatetests, cleantests

    from gpaw.mpi import world
    if parallel:
        from gpaw.mpi import have_mpi
        if not have_mpi:
            # Start again using gpaw-python in parallel:
            arguments = ['mpiexec', '-np', str(parallel),
                         'gpaw-python', '-m', 'asr', 'test'] + sys.argv[2:]
            os.execvp('mpiexec', arguments)

    try:
        generatetests()
        if not tests:
            folder = Path(__file__).parent.parent / 'tests'
            tests = [str(path) for path in folder.glob('test_*.py')]

        if patterns:
            patterns = patterns.split(',')
            tmptests = []
            for pattern in patterns:
                tmptests += [test for test in tests if pattern in test]
            tests = tmptests
        failed = ASRTestRunner(tests, jobs=jobs, show_output=show_output).run()
    finally:
        if world.rank == 0:
            cleantests()

    assert not failed, 'Some tests failed!'


@cli.command()
@click.option('-t', '--tasks', type=str,
              help=('Only choose specific recipes and their dependencies '
                    '(comma separated list of asr.recipes)'),
              default=None)
@click.option('--doforstable',
              help='Only do these recipes for stable materials')
def workflow(tasks, doforstable):
    """Helper function to make workflows for MyQueue"""
    from asr.utils import get_recipes, get_dep_tree

    body = ''
    body += 'from myqueue.task import task\n\n\n'

    isstablefunc = """def is_stable():
    # Example of function that looks at the heat of formation
    # and returns True if the material is stable
    from asr.utils import read_json
    from pathlib import Path
    fname = 'results_convex_hull.json'
    if not Path(fname).is_file():
        return False

    data = read_json(fname)
    if data['hform'] < 0.05:
        return True
    return False\n\n\n"""

    if doforstable:
        body += isstablefunc

    body += 'def create_tasks():\n    tasks = []\n'

    if tasks:
        names = []
        for task in tasks.split(','):
            names += [recipe.name for recipe in get_dep_tree(task)]

    for recipe in get_recipes(sort=True):
        indent = 4
        if tasks:
            if recipe.name not in names:
                continue

        if not recipe.group:
            continue

        if recipe.group not in ['structure', 'property']:
            continue

        if recipe.resources:
            resources = recipe.resources
        else:
            resources = '1:10m'

        if doforstable and recipe.name in doforstable.split(','):
            indent = 8
            body += '    if is_stable():\n'
        body += ' ' * indent + f"tasks += [task('{recipe.name}@{resources}'"
        if recipe.dependencies:
            deps = ','.join(recipe.dependencies)
            body += f", deps='{deps}')"
        else:
            body += ')'
        body += ']\n'

    print(body)

    print('    return tasks')


tests = [{'cli': ['asr run -h']},
         {'cli': ['asr run setup.params asr.relax:ecut 300']},
         {'cli': ['asr run dry setup.params asr.relax:ecut 300']},
         {'cli': ['mkdir folder1',
                  'mkdir folder2',
                  'asr run setup.params asr.relax:ecut'
                  ' 300 in folder1 folder2']},
         {'cli': ['touch str1.json',
                  'asr run shell mv str1.json str2.json']}]
