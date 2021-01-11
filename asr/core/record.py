"""Implement RunSpec and RunRecord."""
import typing
import copy
from .specification import RunSpecification
from .resources import Resources
from .utils import make_property
from .results import get_object_matching_obj_id


# XXX: Change RunRecord name to Record
# XXX: Make MigrationLog object to store migration related info.
# XXX: Remove side_effects object.
# XXX: Make Tags object.

class RunRecord:

    record_version: int = 0
    result = make_property('result')
    side_effects = make_property('side_effects')
    dependencies = make_property('dependencies')
    run_specification = make_property('run_specification')
    resources = make_property('resources')
    migrated_from = make_property('migrated_from')
    migrated_to = make_property('migrated_to')
    migration_id = make_property('migration_id')
    tags = make_property('tags')

    def __init__(  # noqa
            self,
            result: typing.Optional[typing.Any] = None,
            run_specification: typing.Optional[RunSpecification] = None,
            resources: typing.Optional[Resources] = None,
            side_effects: typing.Optional[dict] = None,
            dependencies: typing.Optional[typing.List[str]] = None,
            migration_id: typing.Optional[str] = None,
            migrated_from: typing.Optional[str] = None,
            migrated_to: typing.Optional[str] = None,
            tags: typing.Optional[typing.List[str]] = None,
    ):
        assert type(run_specification) in [RunSpecification, type(None)]
        assert type(resources) in [Resources, type(None)]
        # XXX strictly enforce rest of types.

        self.data = dict(
            run_specification=run_specification,
            result=result,
            resources=resources,
            side_effects=side_effects,
            dependencies=dependencies,
            migration_id=migration_id,
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

    @property
    def name(self):  # noqa
        return self.data['run_specification'].name

    def get_migrations(self, cache):
        """Delegate migration to function objects."""
        obj = get_object_matching_obj_id(self.run_specification.name)
        if obj.migrations:
            return obj.migrations(cache)

    def copy(self):
        data = copy.deepcopy(self.data)
        return RunRecord(**data)

    def __str__(self):  # noqa
        strings = []
        for name, value in self.data.items():
            if name == 'result':
                txt = str(value)
                if len(txt) > 30:
                    strings.append('result=' + str(value)[:30] + '...')
                    continue
            if value is not None:
                strings.append('='.join([str(name), str(value)]))
        return 'Record(' + ', '.join(strings) + ')'

    def __repr__(self):  # noqa
        return self.__str__()

    def __getattr__(self, attr):  # noqa
        if attr in self.data:
            return self.data[attr]
        raise AttributeError

    def __eq__(self, other):  # noqa
        if not isinstance(other, RunRecord):
            return False
        return self.__dict__ == other.__dict__
