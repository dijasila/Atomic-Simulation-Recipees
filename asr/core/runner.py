"""Implement side effect handling."""

from pathlib import Path
from .selector import Selector
from .serialize import JSONSerializer
from .specification import RunSpecification
from .utils import chdir, write_file, read_file
from .root import Repository
from .lock import lock as lock_deco, Lock


serializer = JSONSerializer()

# XXX: This module should probably be called something like work_dir
# or IsolatedDir or WorkingEnv.


def get_workdir_name(
        run_specification: RunSpecification) -> Path:
    name = run_specification.name
    uid = run_specification.uid

    repo = Repository.find_root()
    data_file = repo.asr_path('work_dirs.json')

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

    workdir = repo.asr_path(foldername)
    return workdir


class Runner:

    @property
    def lock(self):
        repo = Repository.find_root()
        return Lock(repo.asr_path('runner.lock'), timeout=10)

    @lock_deco
    def get_workdir_name(self, run_specification):
        return get_workdir_name(run_specification)

    def run(self, func, run_specification):
        workdir = self.get_workdir_name(
            run_specification,
        )

        with chdir(workdir, create=True):
            result = func(run_specification)
        return result

    def make_decorator(
            self,
    ):

        def decorator(func):
            def wrapped(run_specification):
                return self.run(func, run_specification)
            return wrapped

        return decorator

    def __call__(self):
        return self.make_decorator()


runner = Runner()
