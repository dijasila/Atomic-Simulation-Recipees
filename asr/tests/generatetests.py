def flatten(d, parent_key='', sep=':'):
    import collections
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_results(item):
    from asr.utils import read_json
    import numpy as np
    filename = item['file']
    results = flatten(read_json(filename))
    item.pop('file')
    for key, value in item.items():
        ref = value[0]
        precision = value[1]
        assert np.allclose(results[key], ref, atol=precision), \
            f'{filename}[{key}] != {ref} ± {precision}'


def run_test(test):
    import subprocess

    cli = []
    testfunction = None
    fails = False
    results = None

    if 'cli' in test:
        assert isinstance(test['cli'], list), \
            'Type: clitest. Should be a list commands.'
        cli = test['cli']

    if 'test' in test:
        testfunction = test['test']
        assert callable(testfunction), \
            'Function test type should be callable.'

    if 'fails' in test:
        fails = test['fails']

    if 'results' in test:
        results = test['results']

    try:
        for command in cli:
            subprocess.run(command, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           check=True)

        if testfunction:
            testfunction()

        if results:
            for item in results:
                check_results(item)
    except Exception as e:
        if not fails:
            raise AssertionError(e.stderr.decode('ascii'))
    else:
        if fails:
            raise AssertionError('This test should fail but it doesn\'t.')


def make_test_files(module, tests):
    from asr.utils import file_barrier
    from pathlib import Path
    from ase.parallel import world

    for it, test in enumerate(tests):
        assert isinstance(test, dict), f'Unknown Test type {test}'
        name = test.get('name', None)
        if name:
            testname = name + '_gen.py'
        else:
            testname = None

        if not testname:
            id = 0
            while True:
                testname = f'test_{module}_{id}_gen.py'
                if not Path(Path(__file__).parent / testname).exists():
                    break
                id += 1

        text = 'from asr.tests.generatetests import run_test\n'
        text += f'from {module} import tests\n\n\n'
        text += f'run_test(tests[{it}])\n'

        msg = (f'Invalid test name: "{name}". Please name your '
               'tests as "test_{name}".')
        assert testname.startswith('test_'), msg
        filename = Path(__file__).parent / testname
        assert not filename.exists(), \
            f'This file already exists: {filename}'
        with file_barrier(filename):
            if world.rank == 0:
                print(filename)
                filename.write_text(text)


def generatetests():
    # from pathlib import Path
    from asr.utils import get_recipes
    # from ase.parallel import world
    from asr.utils.cli import tests as clitests

    make_test_files('asr.utils.cli', clitests)

    recipes = get_recipes()
    for recipe in recipes:
        if not recipe.main:
            continue
        tests = recipe.main.tests
        if not tests:
            continue
        make_test_files(recipe.name, tests)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_*_gen.py')

    for p in paths:
        p.unlink()


if __name__ == '__main__':
    generatetests()
