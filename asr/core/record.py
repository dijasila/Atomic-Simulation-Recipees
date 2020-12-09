"""Implement RunSpec and RunRecord."""
import typing
from .specification import RunSpecification
from .resources import Resources
from .utils import make_property
from .results import get_object_matching_obj_id


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

    def migrate(self):
        obj = get_object_matching_obj_id(self.run_specification.name)
        if hasattr(obj, 'migrate'):
            record = obj.migrate(self)
        else:
            record = self

        is_migrated = False
        migrated_data = {}
        for attr in [
                'result',
                'run_specification',
                'resources',
                'side_effects',
                'dependencies']:

            attribute = getattr(record, attr)
            if hasattr(attribute, 'migrate'):
                migrated = attribute.migrate()
                if migrated:
                    is_migrated = True
                    migrated_data[attr] = migrated
                else:
                    migrated_data[attr] = attribute
            else:
                migrated_data[attr] = attribute

        if is_migrated:
            return RunRecord(**migrated)

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
