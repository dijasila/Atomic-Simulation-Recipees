"""Lock file implementation.

This was borrowed from ASE.
"""
import os
import errno
import time
import functools


# Only Windows has O_BINARY:
CEW_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0)


def opencew(filename, world=None):
    """Create and open filename exclusively for writing.

    If master cpu gets exclusive write access to filename, a file
    descriptor is returned (a dummy file descriptor is returned on the
    slaves).  If the master cpu does not get write access, None is
    returned on all processors.
    """
    if world is None:
        from ase.parallel import world

    if world.rank == 0:
        try:
            fd = os.open(filename, CEW_FLAGS)
        except OSError as ex:
            error = ex.errno
        else:
            error = 0
            fd = os.fdopen(fd, 'wb')
    else:
        error = 0
        fd = open(os.devnull, 'wb')

    # Syncronize:
    error = world.sum(error)
    if error == errno.EEXIST:
        return None
    if error:
        raise OSError(error, 'Error', filename)
    return fd


class Lock:
    def __init__(self, name='lock', world=None, timeout=float('inf')):
        self.name = name
        self.timeout = timeout
        if world is None:
            from ase.parallel import world
        self.world = world
        self.fd = None
        self.count = 0

    def acquire(self):
        self.count += 1
        if self.count > 1:
            return
        dt = 0.2
        t1 = time.time()
        while True:
            fd = opencew(self.name, self.world)
            if fd is not None:
                self.fd = fd
                break
            time_left = self.timeout - (time.time() - t1)
            if time_left <= 0:
                raise TimeoutError
            time.sleep(min(dt, time_left))
            dt *= 2

    def release(self):
        self.count -= 1
        if self.count > 0:
            return
        self.world.barrier()
        # Important to close fd before deleting file on windows
        # as a WinError would otherwise be raised.
        self.fd.close()
        if self.world.rank == 0:
            os.remove(self.name)
        self.world.barrier()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, tb):
        self.release()


def lock(method):
    """Acquire and release lock file before evaluating wrapped function."""
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.lock is None:
            return method(self, *args, **kwargs)
        else:
            with self.lock:
                return method(self, *args, **kwargs)
    return new_method
