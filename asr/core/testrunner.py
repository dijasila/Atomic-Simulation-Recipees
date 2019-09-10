from asr.core import chdir
import tempfile
from gpaw.cli.info import info
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
        assert 'type' in test, f'No type in test {test}'
        testtype = test['type']
        assert testtype in ['file', 'dict'], f'Unknown test type {testtype}'

        testname = test['name']
        assert 'name' in test, f'No name in test {test}'

        assert testname not in names, f'Duplicate name {testname}'
        names.append(testname)
        if testtype == 'file':
            assert 'path' in test, f'Test has to contain key: path {test}'

            testpath = test['path']
            assert Path(testpath).is_file(), f'Unknown file {testpath}'


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=True):

        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.donetests = []
        self.failed = []
        self.log = stream
        n = 0
        for test in tests:
            n = np.max([n, len(self.get_description(test))])
        self.n = n
        check_tests(self.tests)

    def get_description(self, test):
        testname = test['name']
        testdescription = test.get('description')
        if testdescription:
            description = f'{testname} ({testdescription})'
        else:
            description = f'{testname}'
        return description

    def run(self, raiseexc):
        # Make temporary directory and print some execution info
        tmpdir = tempfile.mkdtemp(prefix='asr-test-')
        info()
        print('Running tests in', tmpdir)
        print(f'Jobs: {self.jobs}')

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
        if raiseexc and self.failed:
            raise AssertionError('Some tests failed!')
        return self.failed

    def run_tests(self):
        for test in self.tests:
            t0 = time.time()
            testname = test['name']
            description = self.get_description(test)
            with chdir(Path(testname), create=True):
                print(f'{description: <{self.n}}', end='', flush=True,
                      file=self.log)
                try:
                    self.run_test(test)
                except Exception:
                    self.failed.append(testname)
                    tb = traceback.format_exc()
                    msg = ('FAILED\n'
                           '{0:#^77}\n'.format('TRACEBACK') +
                           f'{tb}' +
                           '{0:#^77}\n'.format(''))
                    self.write_result(msg, t0)
                    self.donetests.append(testname)
                except KeyboardInterrupt:
                    self.write_result('INTERRUPT', t0)
                else:
                    self.donetests.append(testname)
                    self.write_result('OK', t0)

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
                subprocess.run(command, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               check=True)

            if testfunction:
                testfunction()

            if results:
                for item in results:
                    check_results(item)
        except subprocess.CalledProcessError as e:
            if not fails:
                raise AssertionError(e.stderr.decode('ascii'))
        except Exception:
            if not fails:
                raise
        else:
            if fails:
                raise AssertionError('This test should fail but it doesn\'t.')

    def write_result(self, text, t0):
        t = time.time() - t0
        self.log.write('%10.3f  %s\n' % (t, text))
