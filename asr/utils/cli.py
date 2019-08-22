import click
from click import argument, option


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
            output += f'{row}'
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


@cli.command()
@click.option('-s', '--shell', is_flag=True,
              help='Interpret COMMAND as shell command.')
@click.option('-n', '--not-recipe', is_flag=True,
              help='COMMAND is not a recipe.')
@click.option('-z', '--dry-run', is_flag=True,
              help='Show what would happen without doing anything.')
@click.option('-p', '--parallel', type=int, help='Run on NCORES.',
              metavar='NCORES')
@click.option('-j', '--jobs', type=int,
              help='Run COMMAND in serial on JOBS processes.')
@click.option('-S', '--skip-if-done', is_flag=True,
              help='Skip execution of recipe if done.')
@click.option('--dont-raise', is_flag=True, default=False,
              help='Continue to next folder when encountering error.')
@click.argument('command', nargs=1)
@click.argument('folders', nargs=-1)
def run(shell, not_recipe, dry_run, parallel, command, folders, jobs,
        skip_if_done, dont_raise):
    """Run recipe, python function or shell command in multiple folders.

    Can run an ASR recipe or a shell command. For example, the syntax
    "asr run recipe" will run the relax recipe in the current folder.

    To run a shell script use the syntax 'asr run --shell "echo Hello!"'.
    This example would run "echo Hello!" in the current folder.

    Provide extra arguments to the recipe using 'asr run "recipe --arg1
    --arg2"'.

    Run a recipe in parallel using 'asr run -p NCORES "recipe --arg1"'.

    Run command in multiple folders using "asr run recipe folder1/ folder2/".
    This is also compatible with input arguments to the current command
    through 'asr run "recipe --arg1" folder1/ folder2/'.

    If you dont actually wan't to run the command, i.e., if it is a
    dangerous command, then use the "asr run --dry-run ..." syntax
    where ... could be any of the above commands. For example,
    'asr run --dry-run --shell "echo Hello!" */' would run "echo Hello!"
    in all folders of the current directory.

    Examples:

    \b
    Run the relax recipe:
        asr run relax
    Run the calculate function in the gs module:
        asr run gs@calculate
    Get help for a recipe:
        asr run "relax -h"
    Specify an argument:
        asr run "relax --ecut 600"
    Run a recipe in parallel with an argument:
        asr run -p 2 "relax --ecut 600"
    Run relax recipe in two folders sequentially:
        asr run relax folder1/ folder2/
    Run a shell command in this folder:
        asr run --shell "ase convert gs.gpw structure.json"
    Run a shell command in "folder1/":
        asr run --shell "ase convert gs.gpw structure.json" folder1/
    Don't actually do anything just show what would be done
        asr run --dry-run --shell "mv str1.json str2.json" folder1/ folder2/

    """
    import subprocess
    from pathlib import Path
    from ase.parallel import parprint
    from asr.utils import chdir
    from functools import partial

    assert not (parallel and jobs), '--parallel is incompatible with --jobs'

    prt = partial(parprint, flush=True)

    if parallel:
        assert not shell, \
            ('You cannot execute a shell command in parallel. '
             'Only supported for python modules.')

    if parallel:
        ncores = jobs or parallel
        from gpaw.mpi import have_mpi
        if not have_mpi:
            cmd = f'mpiexec -np {ncores} gpaw-python -m asr run'
            if dry_run:
                cmd += ' --dry-run'
            if jobs:
                cmd += f' --jobs {ncores}'
            if dont_raise:
                cmd += ' --dont-raise'
            cmd += f' {command}" '
            if folders:
                cmd += ' '.join(folders)
            return subprocess.run(cmd, shell=True,
                                  check=True)

    if not folders:
        folders = ['.']
    else:
        prt(f'Number of folders: {len(folders)}')

    nfolders = len(folders)

    if jobs:
        assert jobs <= nfolders, 'Too many jobs and too few folders!'
        for job in range(jobs):
            cmd = 'asr run'
            myfolders = folders[job::jobs]
            if skip_if_done:
                cmd += ' --skip-if-done'
            if dont_raise:
                cmd += ' --dont-raise'
            if shell:
                cmd += ' --shell'
            if dry_run:
                cmd += ' --dry-run'
            cmd += f' "{command}" '
            cmd += ' '.join(myfolders)
            subprocess.Popen(cmd, shell=True)
        return

    # Identify function that should be executed
    if shell:
        command = command.strip()
        if dry_run:
            prt(f'Would run shell command "{command}" '
                f'in {nfolders} folders.')
            return

        for i, folder in enumerate(folders):
            with chdir(Path(folder)):
                prt(f'Running {command} in {folder} ({i + 1}/{nfolders})')
                subprocess.run(command, shell=True)
        return

    # If not shell then we assume that the command is a call
    # to a python module or a recipe
    module, *args = command.split()
    function = None
    if '@' in module:
        module, function = module.split('@')

    if not_recipe:
        assert function, \
            ('If this is not a recipe you have to specify a '
             'specific function to execute.')
    else:
        if not module.startswith('asr.'):
            module = f'asr.{module}'

    import importlib
    mod = importlib.import_module(module)
    if not function:
        function = 'main'
    assert hasattr(mod, function), f'{module}@{function} doesn\'t exist'
    func = getattr(mod, function)

    from asr.utils import ASRCommand
    if isinstance(func, ASRCommand):
        is_asr_command = True
    else:
        is_asr_command = False

    import sys
    if dry_run:
        prt(f'Would run {module}@{function} '
            f'in {nfolders} folders.')
        return

    for i, folder in enumerate(folders):
        with chdir(Path(folder)):
            try:
                if skip_if_done and func.done:
                    continue
                prt(f'In folder: {folder} ({i + 1}/{nfolders})')
                if is_asr_command:
                    func.cli(args=args)
                else:
                    sys.argv = [mod.__name__] + args
                    func()
            except click.Abort:
                break
            except Exception as e:
                if not dont_raise:
                    raise
                else:
                    prt(e)
            except SystemExit:
                print('Unexpected error:', sys.exc_info()[0])
                if not dont_raise:
                    raise


@cli.command()
@click.argument('search', required=False)
def list(search):
    """List and search for recipes.

    If SEARCH is specified: list only recipes containing SEARCH in their
    description."""
    from asr.utils import get_recipes
    recipes = get_recipes()
    recipes.sort(key=lambda x: x.name)
    panel = [['Name', 'Description'],
             ['----', '-----------']]

    for state in ['tested', 'untested']:
        for recipe in recipes:
            if not recipe.state == state.strip():
                continue
            longhelp = recipe._main.__doc__
            if not longhelp:
                longhelp = ''

            shorthelp, *_ = longhelp.split('\n')

            if state == 'untested':
                shorthelp = '(Untested) ' + shorthelp
            if search and (search not in longhelp and
                           search not in recipe.name):
                continue
            status = [recipe.name[4:], shorthelp]
            panel += [status]
        panel += ['\n']

    print(format(panel))


@cli.command()
def status():
    """Show the status of the current folder for all ASR recipes"""
    from asr.utils import get_recipes
    recipes = get_recipes()
    panel = []
    missing_files = []
    for recipe in recipes:
        status = [recipe.name]
        done = recipe.done
        if done:
            if recipe.creates:
                status.append(f'Done -> {recipe.creates}')
            else:
                status.append(f'Done.')
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
         {'cli': ['asr run "setup.params asr.relax:ecut 300"']},
         {'cli': ['asr run --dry-run "setup.params asr.relax:ecut 300"']},
         {'cli': ['mkdir folder1',
                  'mkdir folder2',
                  'asr run "setup.params asr.relax:ecut'
                  ' 300" folder1 folder2']},
         {'cli': ['touch str1.json',
                  'asr run --shell "mv str1.json str2.json"']}]
