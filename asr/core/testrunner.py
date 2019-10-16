from asr.core import chdir
import tempfile
import time
import sys
import traceback
from pathlib import Path
import numpy as np

exclude = []


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
    from pathlib import Path
    from asr.core import read_json

    filename = item['file']
    item.pop('file')
    if not item:
        # Then we just have to check for existence of file:
        assert Path(filename).exists(), f'{filename} doesn\'t exist'
        return
    results = flatten(read_json(filename))
    for key, value in item.items():
        ref = value[0]
        precision = value[1]
        assert np.allclose(results[key], ref, atol=precision), \
            f'{filename}[{key}] != {ref} ± {precision}'


def check_tests(tests):
    names = []
    for test in tests:
        assert isinstance(test, dict), f'Test has to have type dict {test}'
        testname = test['name']
        assert 'name' in test, f'No name in test {test}'

        assert testname not in names, f'Duplicate name {testname}'
        names.append(testname)


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=True):

        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.donetests = []
        self.failed = []
        self.log = stream
        self.n = 77
        check_tests(self.tests)

    def get_description(self, test):
        testname = test['name']
        testdescription = test.get('description')
        if testdescription:
            description = f'{testname} ({testdescription})'
        else:
            description = f'{testname}'
        return description

    def run(self, tmpdir=None):
        # Make temporary directory and print some execution info
        self.cwd = Path('.').absolute()
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(prefix='asr-test-')

        print('Running ASR tests')
        self.log.write('=' * 77 + '\n')
        # if not self.show_output:
        #     sys.stdout = devnull
        ntests = len(self.tests)
        t0 = time.time()

        with chdir(tmpdir):
            self.run_tests()

        sys.stdout = sys.__stdout__
        self.log.write('=' * 77 + '\n')
        ntime = time.time() - t0
        ndone = len(self.donetests)
        self.log.write(f'Ran {ndone} out out {ntests} tests '
                       f'in {ntime:0.1f} seconds\n')
        if self.failed:
            print('Tests failed:', len(self.failed), file=self.log)
        else:
            self.log.write('All tests passed!\n')
        self.log.write('=' * 77 + '\n')
        return self.failed

    def run_tests(self):

        for test in self.tests:
            testname = test['name']
            with chdir(Path(testname), create=True):
                folder = Path('.').absolute()
                print(f'{folder}/', flush=True,
                      file=self.log)
                try:
                    self.run_test(test)
                except Exception:
                    self.failed.append(testname)
                    tb = traceback.format_exc()
                    msg = (' ... FAILED\n'
                           '{0:#^77}\n'.format('TRACEBACK') +
                           f'{tb}' +
                           '{0:#^77}\n'.format(''))
                    print(msg, flush=True, file=self.log)
                    self.donetests.append(testname)
                except KeyboardInterrupt:
                    print(' ... INTERRUPTED', file=self.log)
                    break
                else:
                    self.donetests.append(testname)

    def run_test(self, test):
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
                print(f'    $ {command}', end='', file=self.log, flush=True)
                subprocess.run(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               check=True)
                print(' ... OK', file=self.log, flush=True)

            if testfunction:
                testfunction()

            if results:
                for item in results:
                    check_results(item)
        except subprocess.CalledProcessError as e:
            if fails:
                print(' ... OK', file=self.log, flush=True)
            else:
                raise AssertionError(e.stderr.decode('ascii'))
        except Exception:
            if not fails:
                raise
        else:
            if fails:
                raise AssertionError('This test should fail but it doesn\'t.')

    def write_result(self, text, t0):
        t = time.time() - t0
        self.log.write('\n' + f'{t:10.3f}s {text}'.rjust(77) + '\n')
