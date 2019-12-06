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
@click.argument('search', required=False)
def list(search):
    """List and search for recipes.

    If SEARCH is specified: list only recipes containing SEARCH in their
    description."""
    from asr.core import get_recipes
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
    from asr.core import get_recipes
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
@argument('patterns', nargs=-1, required=False)
@option('-s', '--show-output', is_flag=True,
        help='Show standard output from tests.')
@option('--raiseexc', is_flag=True,
        help='Raise error if tests fail')
@option('--tmpdir', help='Execution dir. If '
        'not specified ASR will decide')
@option('--tag', help='Only run tests with given tag')
@option('--run-coverage', is_flag=True, help='Run coverage module')
def test(patterns, show_output, raiseexc, tmpdir, tag, run_coverage):
    from asr.core.testrunner import TestRunner
    import os
    from pathlib import Path

    # We will log the test home directory if needed
    cwd = Path('.').absolute()

    if run_coverage:
        os.environ['COVERAGE_PROCESS_START'] = str(cwd /
                                                   '.coveragerc')

    def get_tests():
        tests = []

        # Collect tests from recipes
        from asr.core import get_recipes
        recipes = get_recipes()
        for recipe in recipes:
            if recipe.tests:
                id = 0
                for test in recipe.tests:
                    dct = {}
                    dct.update(test)
                    if 'name' not in dct:
                        dct['name'] = f'{recipe.name}_{id}'
                        id += 1
                    tests.append(dct)

        # Get cli tests
        for i, test in enumerate(clitests):
            clitest = {'name': f'clitest_{i}'}
            clitest.update(test)
            tests.append(clitest)

        # Test docstrings
        for recipe in recipes:
            if recipe.__doc__:
                tmptests = doctest(recipe.__doc__)
                for i, test in enumerate(tmptests):
                    test['name'] = f'{recipe.name}_doctest_{i}'
                tests += tmptests
        return tests

    tests = get_tests()

    if patterns:
        tmptests = []
        for test in tests:
            for pattern in patterns:
                if pattern in test['name']:
                    tmptests.append(test)
                    break
        tests = tmptests

    if tag:
        tmptests = []
        for test in tests:
            tags = test.get('tags', [])
            if tag in tags:
                tmptests.append(test)

        tests = tmptests
    failed = TestRunner(tests, show_output=show_output).run(tmpdir=tmpdir)
    if raiseexc and failed:
        raise AssertionError('Some tests failed!')


def doctest(text):
    text = text.split('\n')
    tests = []
    cli = []
    for line in text:
        if not line.startswith(' ' * 8) and cli:
            tests.append({'cli': cli})
            cli = []
        line = line[8:]
        # print(line)
        if line.startswith('$ '):
            cli.append(line[2:])
        elif line.startswith('  ...'):
            cli[-1] += line[5:]
    else:
        if cli:
            tests.append({'cli': cli})

    return tests


@cli.command()
@click.option('-t', '--tasks', type=str,
              help=('Only choose specific recipes and their dependencies '
                    '(comma separated list of asr.recipes)'),
              default=None)
@click.option('--doforstable',
              help='Only do these recipes for stable materials')
def workflow(tasks, doforstable):
    """Helper function to make workflows for MyQueue"""
    from asr.core import get_recipes, get_dep_tree

    body = ''
    body += 'from myqueue.task import task\n\n\n'

    isstablefunc = """def is_stable():
    # Example of function that looks at the heat of formation
    # and returns True if the material is stable
    from asr.core import read_json
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


clitests = [{'cli': ['asr run -h'],
             'tags': ['gitlab-ci']},
            {'cli': ['asr run "setup.params asr.relax:fixcell True"'],
             'tags': ['gitlab-ci']},
            {'cli': ['asr run --dry-run setup.params'],
             'tags': ['gitlab-ci']},
            {'cli': ['mkdir folder1',
                     'mkdir folder2',
                     'asr run setup.params folder1 folder2'],
             'tags': ['gitlab-ci']},
            {'cli': ['touch str1.json',
                     'asr run --shell "mv str1.json str2.json"'],
             'tags': ['gitlab-ci']}]
