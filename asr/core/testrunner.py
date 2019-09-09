from asr.core import chdir
import tempfile
from gpaw.cli.info import info
import time
import sys
import traceback
from pathlib import Path
import numpy as np
import os


exclude = []


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=False):

        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.failed = []
        self.skipped = []
        self.garbage = []
        self.n = max([len(test) for test in tests])

        # Check *all* the files, not just the ones we are supposed to be
        # running right now:
        check_tests()

    def run(self, *args, **kwargs):
        # Make temporary directory
        tmpdir = tempfile.mkdtemp(prefix='asr-test-')
        info()
        print('Running tests in', tmpdir)
        print(f'Jobs: {self.jobs}')

        with chdir(tmpdir):
            self.log.write('=' * 77 + '\n')
            if not self.show_output:
                sys.stdout = devnull
            ntests = len(self.tests)
            t0 = time.time()
            if self.jobs == 1:
                self.run_single()
            else:
                # Run several processes using fork:
                self.run_forked()

            sys.stdout = sys.__stdout__
            self.log.write('=' * 77 + '\n')
            self.log.write('Ran %d tests out of %d in %.1f seconds\n' %
                           (ntests - len(self.tests) - len(self.skipped),
                            ntests, time.time() - t0))
            self.log.write('Tests skipped: %d\n' % len(self.skipped))
            if self.failed:
                print('Tests failed:', len(self.failed), file=self.log)
            else:
                self.log.write('All tests passed!\n')
            self.log.write('=' * 77 + '\n')
        return self.failed

    def run_single(self):
        while self.tests:
            test = self.tests.pop(0)
            try:
                self.run_one(test)
            except KeyboardInterrupt:
                self.tests.append(test)
                break

    def run_forked(self):
        j = 0
        pids = {}
        while self.tests or j > 0:
            if self.tests and j < self.jobs:
                test = self.tests.pop(0)
                pid = os.fork()
                if pid == 0:
                    exitcode = self.run_one(test)
                    os._exit(exitcode)
                else:
                    j += 1
                    pids[pid] = test
            else:
                try:
                    while True:
                        pid, exitcode = os.wait()
                        if pid in pids:
                            break
                except KeyboardInterrupt:
                    for pid, test in pids.items():
                        os.kill(pid, signal.SIGHUP)
                        self.write_result(test, 'STOPPED', time.time())
                        self.tests.append(test)
                    break
                if exitcode == 512:
                    self.failed.append(pids[pid])
                elif exitcode == 256:
                    self.skipped.append(pids[pid])
                del pids[pid]
                j -= 1

    def run_one(self, test):
        exitcode_ok = 0
        exitcode_skip = 1
        exitcode_fail = 2

        if self.jobs == 1:
            self.log.write('%*s' % (-self.n, test))
            self.log.flush()

        t0 = time.time()
        filename = str(test)

        tb = ''
        skip = False

        if test in exclude:
            self.register_skipped(test, t0)
            return exitcode_skip
        
        assert test.endswith('.py')
        dirname = Path(test).with_suffix('').name
        if os.path.isabs(dirname):
            mydir = os.path.split(__file__)[0]
            dirname = os.path.relpath(dirname, mydir)

        # We don't want files anywhere outside the tempdir.
        assert not dirname.startswith('../')  # test file outside sourcedir

        if mpi.rank == 0:
            os.makedirs(dirname)
            (Path(dirname) / Path(filename).name).write_text(
                Path(filename).read_text())
        mpi.world.barrier()
        cwd = os.getcwd()
        os.chdir(dirname)

        try:
            setup_paths[:] = self.setup_paths
            loc = {}
            with open(filename) as fd:
                exec(compile(fd.read(), filename, 'exec'), loc)
            loc.clear()
            del loc
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except ImportError as ex:
            if sys.version_info[0] >= 3:
                module = ex.name
            else:
                module = ex.args[0].split()[-1].split('.')[0]
            if module == 'scipy':
                skip = True
            else:
                tb = traceback.format_exc()
        except AttributeError as ex:
            if (ex.args[0] ==
                "'module' object has no attribute 'new_blacs_context'"):
                skip = True
            else:
                tb = traceback.format_exc()
        except Exception:
            tb = traceback.format_exc()
        finally:
            os.chdir(cwd)

        mpi.ibarrier(timeout=60.0)  # guard against parallel hangs

        me = np.array(tb != '')
        everybody = np.empty(mpi.size, bool)
        mpi.world.all_gather(me, everybody)
        failed = everybody.any()
        skip = mpi.world.sum(int(skip))

        if failed:
            self.fail(test, np.argwhere(everybody).ravel(), tb, t0)
            exitcode = exitcode_fail
        elif skip:
            self.register_skipped(test, t0)
            exitcode = exitcode_skip
        else:
            self.write_result(test, 'OK', t0)
            exitcode = exitcode_ok

        return exitcode

    def register_skipped(self, test, t0):
        self.write_result(test, 'SKIPPED', t0)
        self.skipped.append(test)

    def check_garbage(self):
        gc.collect()
        n = len(gc.garbage)
        self.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], self.garbage))

    def fail(self, test, ranks, tb, t0):
        if mpi.size == 1:
            text = 'FAILED!\n%s\n%s%s' % ('#' * 77, tb, '#' * 77)
            self.write_result(test, text, t0)
        else:
            tbs = {tb: [0]}
            for r in range(1, mpi.size):
                if mpi.rank == r:
                    mpi.send_string(tb, 0)
                elif mpi.rank == 0:
                    tb = mpi.receive_string(r)
                    if tb in tbs:
                        tbs[tb].append(r)
                    else:
                        tbs[tb] = [r]
            if mpi.rank == 0:
                text = ('FAILED! (rank %s)\n%s' %
                        (','.join([str(r) for r in ranks]), '#' * 77))
                for tb, ranks in tbs.items():
                    if tb:
                        text += ('\nRANK %s:\n' %
                                 ','.join([str(r) for r in ranks]))
                        text += '%s%s' % (tb, '#' * 77)
                self.write_result(test, text, t0)

        self.failed.append(test)

    def write_result(self, test, text, t0):
        t = time.time() - t0
        if self.jobs > 1:
            self.log.write('%*s' % (-self.n, test))
        self.log.write('%10.3f  %s\n' % (t, text))
