
def run_test(test):
    import subprocess

    cli = []
    testfunction = None
    fail = False

    if 'cli' in test:
        assert isinstance(test['cli'], list), \
            'Type: clitest. Should be a list commands.'
        cli = test['cli']

    if 'test' in test:
        assert callable(testfunction), \
            'Function test type should be callable.'
        testfunction = test['test']

    if 'fail' in test:
        fail = test['fail']

    try:
        for command in cli:
            subprocess.run(command, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           check=True)

        if testfunction:
            testfunction()
    except Exception:
        if not fail:
            raise
    else:
        if fail:
            raise AssertionError('This test should fail but it doesn\'t.')


def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes, file_barrier
    from ase.parallel import world

    recipes = get_recipes()

    for recipe in recipes:
        if not hasattr(recipe.main, 'tests'):
            continue
        tests = recipe.main.tests
        if not tests:
            continue
        testnames = []
        for it, test in enumerate(tests):
            testname = None

            if callable(test):
                testname = test.__name__ + '_gen.py'
            else:
                assert isinstance(test, dict), ('Unknown Test type in '
                                                f'{recipe.name}: {test}')
                name = test.get('name', None)
                if name:
                    testname = name + '_gen.py'
                else:
                    testname = None

            if not testname:
                id = 0
                while True:
                    testname = f'test_{recipe.name}_{id}_gen.py'
                    if testname not in testnames:
                        break
                    id += 1

            text = 'from asr.tests.generatetests import run_test\n'
            text += f'from {recipe.name} import tests\n\n\n'
            text += f'run_test(tests[{it}])'

            msg = (f'Invalid test name: "{name}". Please name your '
                   'tests as "test_{name}".')
            assert testname.startswith('test_'), msg
            assert testname not in testnames, \
                f'Duplicate test name:{name}!'
            testnames.append(testname)
            filename = Path(__file__).parent / testname
            with file_barrier(filename):
                if world.rank == 0:
                    filename.write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_*_gen.py')

    for p in paths:
        p.unlink()


if __name__ == '__main__':
    generatetests()
