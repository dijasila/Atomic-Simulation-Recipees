def generatetests():
    from pathlib import Path
    from asr.utils import get_recipes

    recipes = get_recipes()

    for recipe in recipes:
        if not hasattr(recipe.main, 'tests'):
            continue
        tests = recipe.main.tests
        if not tests:
            continue
        testnames = []
        for test in tests:
            testname = None
            cli = None
            testfunction = None
            fail = False

            if callable(test):
                testfunction = test
                testname = test.__name__ + '_gen.py'
            else:
                assert isinstance(test, dict), ('Unknown Test type in '
                                                f'{recipe.name}: {test}')
                name = test.get('name', None)
                if name:
                    testname = name + '_gen.py'
                else:
                    testname = None
                cli = test.get('cli', None)
                testfunction = test.get('test', None)
                fail = test.get('fail', False)

            if not testname:
                id = 0
                while True:
                    testname = f'test_{recipe.name}_{id}_gen.py'
                    if testname not in testnames:
                        break
                    id += 1

            text = ''

            if cli:
                assert isinstance(test['cli'], list), \
                    'Type: clitest. Should be a list commands.'
                text += 'import subprocess\n\n\n'
                text += 'def clitest():\n'
                commands = []
                for command in test['cli']:
                    parts = command.split()
                    string = ', '.join([f"'{part}'" for part in parts])
                    commands.append(string)

                for command in commands:
                    text += f'    subprocess.run([{command}])\n\n\n'

                if fail:
                    indent = 4
                    text += 'try:\n'
                else:
                    indent = 0

                text += ' ' * indent + 'clitest()\n'

            if testfunction:
                assert callable(testfunction), \
                    'Function test type should be callable.'
                testfunctionname = {testfunction.__name__}
                text = f'from module import {testfunctionname}\n' + text
                text += ' ' * indent + f'{testfunctionname}()\n'

            if fail:
                text += ('except Exception:\n'
                         '    exit()\n'
                         'assert False')
            msg = (f'Invalid test name: "{name}". Please name your '
                   'tests as "test_{name}".')
            assert testname.startswith('test_'), msg
            assert testname not in testnames, \
                f'Duplicate test name:{name}!'
            testnames.append(testname)
            print(f'Writing {testname}')
            (Path(__file__).parent / testname).write_text(text)


def cleantests():
    from pathlib import Path

    paths = Path(__file__).parent.glob('test_*_gen.py')

    for p in paths:
        p.unlink()


if __name__ == '__main__':
    generatetests()
