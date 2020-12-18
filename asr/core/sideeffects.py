"""Implement side effect handling."""

import os
import pathlib
import shutil
import typing
from .serialize import JSONSerializer
from hashlib import sha256
from asr.core.utils import sha256sum
from .specification import RunSpecification
from .utils import chdir


class SideEffect:  # noqa

    def __init__(self, filename, path):  # noqa
        self.filename = filename
        self.path = path
        self.hashes = {'sha256': sha256sum(filename)}

    def __str__(self):  # noqa
        return f'SideEffect({self.filename})'

    def __fspath__(self):  # noqa
        return self.path

    def restore(self):  # noqa
        pathlib.Path(self.filename).write_bytes(
            pathlib.Path(self.path).read_bytes())


side_effects_stack = []


def move_files(mapping: typing.Dict[str, SideEffect]):  # noqa

    for filename, side_effect in mapping.items():

        path = pathlib.Path(filename)
        final_filename = side_effect.path
        directory = pathlib.Path(final_filename).parent
        if not directory.is_dir():
            os.makedirs(directory)
        path.rename(final_filename)


class RegisterSideEffects():  # noqa

    def __init__(self, side_effects_stack=side_effects_stack,  # noqa
                 serializer=JSONSerializer(), hash_func=sha256):
        self.side_effects_stack = side_effects_stack
        self._root_dir = None
        self.serializer = serializer
        self.hash_func = hash_func

    def get_hash_of_run_spec(self, run_spec):  # noqa
        return self.hash_func(
            self.serializer.serialize(
                run_spec
            ).encode()
        ).hexdigest()

    def get_workdir_name(self, root_dir,  # noqa
                         run_specification: RunSpecification) -> pathlib.Path:
        hsh = self.get_hash_of_run_spec(run_specification)
        workdir = root_dir / f'.asr/{run_specification.name}{hsh[:8]}'
        return workdir

    def chdir_to_root_dir(self):  # noqa
        if self._root_dir:
            os.chdir(self._root_dir)

    def restore_to_previous_workdir(self):  # noqa
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

    def get_side_effect_name(self, filename, uid):  # noqa
        side_effect_destination_dir = pathlib.Path(
            self._root_dir / '.asr/side_effects').absolute()
        return str(side_effect_destination_dir / (uid[:15] + filename))

    def make_decorator(  # noqa
            self,
    ):

        def decorator(func):
            def wrapped(asrcontrol, run_specification):
                current_dir = pathlib.Path().absolute()
                if self._root_dir is None:
                    self._root_dir = current_dir

                workdir = self.get_workdir_name(
                    self._root_dir, run_specification)
                with self as frame:
                    def register_side_effect(filename):
                        return self.register_single_side_effect(
                            filename,
                            run_specification.uid
                        )
                    asrcontrol.register_side_effect = register_side_effect
                    with chdir(workdir, create=True):
                        frame['workdir'] = workdir
                        run_record = func(asrcontrol, run_specification)
                        move_files(
                            frame['side_effects'],
                        )

                    # shutil.rmtree(workdir)
                    run_record.side_effects = frame['side_effects']

                if not self.side_effects_stack:
                    self._root_dir = None
                return run_record
            return wrapped

        return decorator

    def register_single_side_effect(self, filename, uid):  # noqa
        frame = self.side_effects_stack[-1]
        name = self.get_side_effect_name(
            filename, uid
        )
        side_effect = SideEffect(filename, name)
        frame['side_effects'][filename] = side_effect
        return side_effect

    def __call__(self):  # noqa
        return self.make_decorator()


register_side_effects = RegisterSideEffects()
