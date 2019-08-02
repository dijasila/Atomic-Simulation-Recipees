from gpaw import setup_paths
from gpaw.test import TestRunner
from asr.utils import chdir
import tempfile
from gpaw import mpi
from gpaw.cli.info import info
import time
import sys
import traceback
from pathlib import Path
import numpy as np
import os


exclude = []


class ASRTestRunner(TestRunner):
    def __init__(self, *args, **kwargs):
        TestRunner.__init__(self, *args, **kwargs)

    def run(self, *args, **kwargs):
        # Make temporary directory
        if mpi.rank == 0:
            tmpdir = tempfile.mkdtemp(prefix='asr-test-')
        else:
            tmpdir = None
        tmpdir = mpi.broadcast_string(tmpdir)
        if mpi.rank == 0:
            info()
            print('Running tests in', tmpdir)
            print('Jobs: {}, Cores: {}'
                  .format(self.jobs, mpi.size))

        with chdir(tmpdir):
            failed = TestRunner.run(self, *args, **kwargs)
        
        return failed

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
