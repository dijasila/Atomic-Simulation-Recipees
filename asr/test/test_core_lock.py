import os
import pytest
from asr.core.lock import Lock
from asr.core.filetype import ASRPath


@pytest.fixture(params=['lockfile', ASRPath('lockfile')])
def lockfile(request, asr_tmpdir):
    pth = request.param
    return pth


@pytest.mark.ci
def test_same_process_can_acquire_lock_twice(lockfile):
    """Test timeout on Lock.acquire()."""
    lock = Lock(lockfile, timeout=0.3)
    with lock:
        with lock:
            ...


@pytest.mark.ci
def test_lock_close_file_descriptor(lockfile):
    """Test that lock file descriptor is properly closed."""
    # The choice of timeout=1.0 is arbitrary but we don't want to use
    # something that is too large since it could mean that the test
    # takes long to fail.
    lock = Lock(lockfile, timeout=1.0)
    with lock:
        pass

    # If fstat raises OSError this means that fd.close() was
    # not called.
    with pytest.raises(OSError):
        os.fstat(lock.fd.name)
