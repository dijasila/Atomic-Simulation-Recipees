"""Implement side effect handling."""

import pathlib
from .selector import Selector
from .serialize import JSONSerializer
from .specification import RunSpecification
from .utils import chdir, write_file, read_file
from .root import root_is_initialized
from .filetype import ASRPath


serializer = JSONSerializer()

# XXX: This module should probably be called something like work_dir
# or IsolatedDir or WorkingEnv.


def get_workdir_name(
        run_specification: RunSpecification) -> pathlib.Path:
    name = run_specification.name
    uid = run_specification.uid
    assert root_is_initialized()
    data_file = ASRPath('work_dirs.json')
    if not data_file.is_file():
        work_dirs = {}
        write_file(data_file, serializer.serialize(work_dirs))
    else:
        work_dirs = serializer.deserialize(read_file(data_file))

    sel = Selector()
    sel.parameters = sel.EQ(run_specification.parameters)
    sel.name = sel.EQ(run_specification.name)

    for foldername, other_run_spec in work_dirs.items():
        if sel.matches(other_run_spec):
            break
    else:
        foldername = f'{name}-{uid[:10]}'
        work_dirs[foldername] = run_specification
        write_file(data_file, serializer.serialize(work_dirs))

    workdir = ASRPath(foldername)
    return workdir


class IsolatedWorkDir():

    def make_decorator(
            self,
    ):

        def decorator(func):
            def wrapped(run_specification):
                workdir = get_workdir_name(
                    run_specification,
                )

                with chdir(workdir, create=True):
                    run_record = func(run_specification)

                return run_record
            return wrapped

        return decorator

    def __call__(self):
        return self.make_decorator()


isolated_work_dir = IsolatedWorkDir()
