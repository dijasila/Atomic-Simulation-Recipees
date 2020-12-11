"""Implement RunSpec and RunRecord."""
import typing
import copy
from .specification import RunSpecification
from .resources import Resources
from .utils import make_property
from .results import get_object_matching_obj_id
from .migrate import is_migratable


class RunRecord:  # noqa

    record_version: int = 0
    result = make_property('result')
    side_effects = make_property('side_effects')
    dependencies = make_property('dependencies')
    run_specification = make_property('run_specification')
    resources = make_property('resources')
    migrated_from = make_property('migrated_from')
    migrated_to = make_property('migrated_to')
    migrations = make_property('migrations')
    tags = make_property('tags')

    def __init__(  # noqa
            self,
            result: typing.Any,
            run_specification: RunSpecification = None,
            resources: Resources = None,
            side_effects: 'SideEffects' = None,
            dependencies: typing.List[str] = None,
            migrations: typing.List[str] = None,
            migrated_from: str = None,
            migrated_to: str = None,
            tags: typing.List[str] = None,
    ):
        assert type(run_specification) == RunSpecification
        assert type(resources) == Resources
        if migrations is None:
            migrations = []
        # XXX strictly enforce rest of types.
        self.data = dict(
            run_specification=run_specification,
            result=result,
            resources=resources,
            side_effects=side_effects,
            dependencies=dependencies,
            migrations=migrations,
            migrated_from=migrated_from,
            migrated_to=migrated_to,
            tags=tags,
        )

    @property
    def parameters(self):  # noqa
        return self.data['run_specification'].parameters

    @property
    def uid(self):  # noqa
        return self.data['run_specification'].uid

    def migrate(self, cache):
        """Delegate migration to function objects."""
        obj = get_object_matching_obj_id(self.run_specification.name)
        if is_migratable(obj):
            obj.migrate(cache)

    def copy(self):
        data = copy.deepcopy(self.data)
        return RunRecord(**data)

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
