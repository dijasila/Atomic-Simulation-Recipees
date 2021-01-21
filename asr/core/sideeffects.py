"""Implement side effect handling."""

import os
import pathlib
from .specification import RunSpecification
from .utils import chdir


# XXX: This module should probably be called something like work_dir
# or IsolatedDir or WorkingEnv.

side_effects_stack = []


class RegisterSideEffects():

    def __init__(self, side_effects_stack=side_effects_stack):
        self.side_effects_stack = side_effects_stack
        self._root_dir = None

    def get_workdir_name(self, root_dir,
                         run_specification: RunSpecification) -> pathlib.Path:
        name = run_specification.name
        uid = run_specification.uid
        workdir = root_dir / f'.asr/{name}-{uid[:10]}'
        return workdir

    def chdir_to_root_dir(self):
        if self._root_dir:
            os.chdir(self._root_dir)

    def restore_to_previous_workdir(self):
        if self.side_effects_stack:
            os.chdir(self.side_effects_stack[-1]['workdir'])

    def __enter__(self):
        """Append empty side effect object to stack."""
        frame = {
            'side_effects': {},
            'clean_files': [],
            'workdir': None,
        }

        self.side_effects_stack.append(frame)
        return frame

    def __exit__(self, type, value, traceback):
        """Register side effects and pop side effects from stack."""
        frame = self.side_effects_stack[-1]
        for filename in frame['clean_files']:
            pathlib.Path(filename).unlink()
        self.side_effects_stack.pop()

    def make_decorator(
            self,
    ):

        def decorator(func):
            def wrapped(run_specification):
                current_dir = pathlib.Path().absolute()
                if self._root_dir is None:
                    self._root_dir = current_dir

                workdir = self.get_workdir_name(
                    self._root_dir,
                    run_specification,
                )
                with self as frame:

                    with chdir(workdir, create=True):
                        frame['workdir'] = workdir
                        run_record = func(run_specification)

                    # shutil.rmtree(workdir)
                    # run_record.side_effects = frame['side_effects']

                if not self.side_effects_stack:
                    self._root_dir = None
                return run_record
            return wrapped

        return decorator

    def __call__(self):
        return self.make_decorator()


register_side_effects = RegisterSideEffects()
