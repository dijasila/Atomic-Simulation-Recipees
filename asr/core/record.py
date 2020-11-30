"""Implement RunSpec and RunRecord."""
import typing
from .specification import RunSpecification
from .resources import Resources
from .utils import make_property


class RunRecord:  # noqa

    record_version: int = 0
    result = make_property('result')
    side_effects = make_property('side_effects')
    dependencies = make_property('dependencies')
    run_specification = make_property('run_specification')
    resources = make_property('resources')

    def __init__(  # noqa
            self,
            result: typing.Any,
            run_specification: RunSpecification = None,
            resources: 'Resources' = None,
            side_effects: 'SideEffects' = None,
            dependencies: typing.List[str] = None,
    ):
        self.data = dict(
            run_specification=run_specification,
            result=result,
            resources=resources,
            side_effects=side_effects,
            dependencies=dependencies,
        )

    @property
    def parameters(self):  # noqa
        return self.data['run_specification'].parameters

    @property
    def uid(self):  # noqa
        return self.data['run_specification'].uid

    def __str__(self):  # noqa
        string = str(self.run_specification)
        maxlength = 25
        if len(string) > maxlength:
            string = string[:maxlength] + '...'
        return f'RunRec({string})'

    def __repr__(self):  # noqa
        return self.__str__()

    def __getstate__(self):  # noqa
        return self.__dict__

    def __setstate__(self, state):  # noqa
        self.__dict__.update(state)

    def __hash__(self):  # noqa
        return hash(str(self.run_specification))

    def __getattr__(self, attr):  # noqa
        if attr in self.data:
            return self.data[attr]
        raise AttributeError

    def __eq__(self, other):  # noqa
        if not isinstance(other, RunRecord):
            return False
        return hash(self) == hash(other)
