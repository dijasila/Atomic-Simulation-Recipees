def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes
    import inspect

    recipes = get_recipes()

    for recipe in recipes:
        if not hasattr(recipe.main, 'tests'):
            continue
        tests = recipe.main.tests
        if not tests:
            continue
        testnames = []
        for test in tests:
            name = None
            cli = None
            testfunction = None

            if callable(test):
                testfunction = test
                name = test.__name__

            else:
                assert isinstance(test, dict), ('Unknown Test type in '
                                                f'{recipe.name}: {test}')
                name = test.get('name', None)
                cli = test.get('cli', None)
                testfunction = test.get('test', None)

            assert name, ('You must give your test a name! '
                          f'{recipe.name}: {test}')
            testname = f'{name}.py'
            text = ''

            if cli:
                assert isinstance(test['cli'], list), \
                    'Type: clitest. Should be a list commands.'
                text += 'import subprocess\n\n'
                commands = []
                for command in test['cli']:
                    parts = command.split()
                    string = ', '.join([f"'{part}'" for part in parts])
                    commands.append(string)

                for command in commands:
                    text += f'subprocess.run([{command}])\n'

            if testfunction:
                assert callable(testfunction), \
                    'Function test type should be callable.'
                text += '\n' * 2 + inspect.getsource(testfunction) + '\n' * 2
                text += testfunction.__name__ + '()\n'

            msg = (f'Invalid test name: "{name}". Please name your '
                   'tests as "test_{name}".')
            assert name.startswith('test_'), msg

            assert testname not in testnames, \
                f'Duplicate test name:{name}!'
            print(f'Writing {testname}')
            (Path(__file__).parent / testname).write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_auto_*.py')

    for p in paths:
        p.unlink()


if __name__ == '__main__':
    generatetests()
