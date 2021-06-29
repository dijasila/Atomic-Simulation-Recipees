from .specification import RunSpecification
import typing
import time
import contextlib
from dataclasses import dataclass


@dataclass
class Resources:  # noqa

    execution_start: typing.Optional[float] = None
    execution_end: typing.Optional[float] = None
    execution_duration: typing.Optional[float] = None
    ncores: typing.Optional[int] = None

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            lines.append(f'{key}={value}')
        return '\n'.join(lines)


@contextlib.contextmanager
def _register_resources(run_specification: RunSpecification):
    from ase.parallel import world
    execution_start = time.time()
    resources = Resources()
    yield resources
    execution_end = time.time()
    resources.execution_start = execution_start
    resources.execution_end = execution_end
    resources.execution_duration = execution_end - execution_start
    resources.ncores = world.size


def register_resources():  # noqa
    def wrapper(func):
        def wrapped(run_specification):
            with _register_resources(run_specification) as resources:
                run_record = func(run_specification)
            run_record.resources = resources
            return run_record
        return wrapped
    return wrapper
