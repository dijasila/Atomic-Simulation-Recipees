"""Implements record migration functionality."""
import abc
import typing
import collections.abc
from dataclasses import dataclass, field
from .command import get_recipes
from .selector import Selector
from .record import Record
from .specification import get_new_uuid
from .history import History


class UnapplicableMigration(Exception):
    """Raise when migration doesn't apply."""


RecordUID = str
UID = str


class Modification(abc.ABC):
    """Class that represents a record modification."""

    @abc.abstractmethod
    def apply(self, record: Record) -> Record:
        pass

    @abc.abstractmethod
    def revert(self, record: Record) -> Record:
        pass

    def __call__(self, record: Record) -> Record:
        return self.apply(record)


@dataclass
class DumbModification(Modification):
    """A very simple implementation of a record modification."""

    previous_record: Record
    new_record: Record

    def apply(self, record):
        return self.new_record

    def revert(self, record):
        return self.previous_record


@dataclass
class DiffModification(Modification):
    differences: typing.List['Difference'] = field(default_factory=list)

    def apply(self, record: Record):
        for difference in self.differences:
            difference.apply(record)

    def revert(self, record: Record):
        for difference in self.differences:
            difference.revert(record)


@dataclass
class Difference(abc.ABC):
    """Class that represent a single attribute difference between records."""

    attribute: 'AttributeSequence'

    @abc.abstractmethod
    def apply(self, obj: typing.Any):
        ...

    @abc.abstractmethod
    def revert(self, obj: typing.Any):
        ...


@dataclass
class NewAttribute(Difference):

    value: typing.Any

    def apply(self, obj: typing.Any):
        self.attribute.set(obj, self.value)

    def revert(self, obj: typing.Any):
        self.attribute.delete(obj)


@dataclass
class DeletedAttribute(Difference):

    value: typing.Any

    def apply(self, obj: typing.Any):
        self.attribute.delete(obj)

    def revert(self, obj: typing.Any):
        self.attribute.set(obj, self.value)


@dataclass
class NewValue(Difference):
    new_value: typing.Any
    old_value: typing.Any

    def apply(self, obj: typing.Any):
        self.attribute.set(obj, self.new_value)

    def revert(self, obj: typing.Any):
        self.attribute.set(obj, self.old_value)


@dataclass
class Attribute:
    """Class that represents an object attribute."""

    name: str

    def set(self, obj, value):
        setattr(obj, self.name, value)

    def get(self, obj):
        getattr(obj, self.name)

    def delete(self, obj):
        delattr(obj, self.name)


@dataclass
class Item:
    """Class that represents an object item."""

    name: str

    def set(self, obj, value):
        obj[self.name] = value

    def get(self, obj):
        return obj[self.name]

    def delete(self, obj):
        del obj[self.name]


@dataclass
class AttributeSequence:
    attrs: typing.List[typing.Union['Attribute', 'Item']]

    def set(self, obj, value):
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].set(obj, value)

    def get(self, obj):
        for attr in self.attrs:
            obj = attr.get(obj)
        return obj

    def delete(self, obj):
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].delete(obj)

    def __getitem__(self, item):
        return AttributeSequence(self.attrs[item])

    def __add__(self, other):
        return AttributeSequence(self.attrs + other.attrs)


def make_modification(old_record: Record, new_record: Record):
    """Search for differences between objects and make resulting modification."""
    differences = get_differences(old_record, new_record)
    return DiffModification(differences)


def get_differences(obj1, obj2, prepend: typing.Optional[AttributeSequence] = None):
    if prepend is None:
        prepend = AttributeSequence()
    differences = []
    attrs_and_values1 = get_attributes_and_values(obj1)
    attrs_and_values2 = get_attributes_and_values(obj2)
    attrs1 = set(attrs_and_values1)
    attrs2 = set(attrs_and_values2)
    deleted_attrs = attrs1 - attrs2
    new_attrs = attrs2 - attrs2

    for attr in deleted_attrs:
        differences.append(
            DeletedAttribute(
                attribute=prepend + attr,
                old=attrs_and_values1[attr],
            )
        )

    for attr in new_attrs:
        differences.append(
            NewAttribute(
                attribute=prepend + attr,
                new=attrs_and_values2[attr]
            )
        )
    common_attrs = attrs1.intersection(attrs2)
    for attr in common_attrs:
        value1 = attrs_and_values1[attr]
        value2 = attrs_and_values2[attr]
        if type(value1) != type(value2):
            differences.append(
                NewValue(
                    attributes=prepend + attr,
                    old=value1,
                    new=value2,
                )
            )
            continue
        diffs_inside_values = get_differences(value1, value2, prepend=attr)
        differences.extend(diffs_inside_values)

    return differences


def get_attributes_and_values(obj):
    attributes_and_values = {}
    if isinstance(obj, collections.abc.Iterable):
        for name in obj:
            attributes_and_values[AttributeSequence([Item(name)])] = obj[name]
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            attributes_and_values[AttributeSequence([Attribute(key)])] = value
    elif hasattr(obj, '__slots__'):
        for key in obj.__slots__:
            value = getattr(obj, key)
            attributes_and_values[AttributeSequence([Attribute(key)])] = value

    return attributes_and_values


@dataclass
class Revision:
    """Container for logging migration information."""

    migration_uid: typing.Optional[UID]
    uid: UID
    description: str
    modification: Modification

    def apply(self, record):
        record.history.add(self)


@dataclass
class RevisionHistory(History):
    """A class the represents the revision history."""

    history: typing.List[Revision] = field(default_factory=list)

    def add(self, revision: Revision):
        self.history.append(revision)

    @property
    def latest_revision(self) -> typing.Union[None, UID]:
        """Get the latest revision, 'None' if no revisions."""
        if not self.history:
            return None
        latest_revision = self.history[-1]
        return latest_revision


@dataclass
class Migration:
    """A class to update a record to a greater version."""

    function: typing.Callable
    description: str
    uid: typing.Optional[None] = None
    eagerness: int = 0

    def apply(self, record: Record) -> Record:
        """Apply migration to record and return mutated record."""
        migrated_record = self.function(record.copy())
        modification = make_modification(
            old_record=record, new_record=migrated_record)
        revision = Revision(
            description=self.description,
            modification=modification,
            migration_uid=self.uid,
            uid=get_new_uuid(),
        )

        return revision

    def __call__(self, record: Record) -> Record:
        return self.apply(record)

    def __str__(self):
        return self.description


class MakeMigrations(abc.ABC):
    """Abstract Base class for factory for making migrations."""

    @abc.abstractmethod
    def make_migrations(self, record: Record) -> typing.List[Migration]:
        pass

    def __call__(self, record: Record) -> typing.List[Migration]:
        return self.make_migrations(record)


@dataclass
class SelectorMigrationGenerator(MakeMigrations):

    migration: Migration
    selector: Selector = field(default_factory=Selector)

    def make_migrations(self, record: Record) -> typing.List[Migration]:
        """Check if migration applies to record."""
        is_match = self.selector.matches(record)
        if is_match:
            return [self.migration]
        else:
            return []


@dataclass
class GeneralMigrationGenerator(MakeMigrations):

    applies: typing.Callable
    migration: Migration

    def make_migrations(self, record: Record) -> typing.List[Migration]:
        if self.applies(record):
            return [self.migration]
        return []


@dataclass
class CollectionMigrationGenerator(MakeMigrations):
    """Generates migrations from a collection of migration generators."""

    migration_generators: typing.List[MakeMigrations]

    def extend(self, migration_generators: typing.List[MakeMigrations]):
        self.migration_generators = self.migration_generators + migration_generators

    def make_migrations(self, record: Record) -> typing.List[Migration]:
        migrations = [
            migration
            for make_migrations in self.migration_generators
            for migration in make_migrations(record)
        ]

        return migrations


@dataclass
class RecordMigration:
    """A class that represents a record migration."""

    initial_record: Record
    migrated_record: Record
    revisions: typing.List[Revision]
    errors: typing.List[typing.Tuple[Migration, Exception]]

    def has_revisions(self):
        """Has migrations to apply."""
        return bool(self.revisions)

    def has_errors(self):
        """Has failed migrations."""
        return bool(self.errors)

    def apply(self, cache):
        """Apply record migration to a cache."""
        cache.update(self.migrated_record)

    def __bool__(self):
        return self.has_revisions()

    def __str__(self):
        nrev = len(self.revisions)
        nerr = len(self.errors)
        revisions_string = ' -> '.join([
            str(migration) for migration in self.revisions])
        problem_string = (
            ', '.join(f'{mig} err="{err}"' for mig, err in self.errors)
        )
        return (
            f'UID=#{self.initial_record.uid[:8]} '
            + (f'name={self.initial_record.name}. ')
            + (f'{nrev} revision(s). ' if nrev > 0 else '')
            + (f'{nerr} migration error(s)! ' if self.errors else '')
            + (f'{revisions_string}. ' if nrev > 0 else '')
            + (f'{problem_string}.' if self.errors else '')
        )


def make_record_migration(
    record: Record,
    migration_generator: MakeMigrations,
) -> RecordMigration:
    """Construct a record migration."""
    migrated_record = record.copy()
    applied_migrations = []
    problematic_migrations = []
    errors = []
    revisions = []
    while True:
        applicable_migrations = migration_generator(migrated_record)
        candidate_migrations = [
            mig for mig in applicable_migrations
            if (
                mig not in problematic_migrations
                and mig not in applied_migrations
            )
        ]
        if not candidate_migrations:
            break

        migration = max(candidate_migrations, key=lambda mig: mig.eagerness)
        try:
            revision = migration(migrated_record)
            migrated_record = revision.apply(migrated_record)
        except Exception as err:
            problematic_migrations.append(migration)
            errors.append((migration, err))
            continue
        applied_migrations.append(migration)
    return RecordMigration(
        initial_record=record,
        revisions=revisions,
        migrated_record=migrated_record,
        errors=errors,
    )


def get_instruction_migration_generator() -> CollectionMigrationGenerator:
    """Collect record migrations from all recipes."""
    recipes = get_recipes()
    migrations = CollectionMigrationGenerator(migration_generators=[])
    for recipe in recipes:
        if recipe.migrations:
            migrations.extend(recipe.migrations)

    return migrations


def make_migration_generator(
    type='selector',
    **kwargs,
) -> MakeMigrations:
    if type == 'selector':
        return make_selector_migration_generator(**kwargs)
    else:
        raise NotImplementedError


def make_selector_migration_generator(
    *,
    selector,
    uid,
    function,
    description=None,
    eagerness=0,
):
    if isinstance(selector, dict):
        selector = Selector(
            **{key: Selector.EQ(value) for key, value in selector.items()})
    if description is None:
        description = function.__doc__.splitlines()[0]
    mig = Migration(
        function=function,
        uid=uid,
        description=description,
        eagerness=eagerness,
    )
    return SelectorMigrationGenerator(selector=selector, migration=mig)


def migration(
    function=None,
    *,
    selector=None,
    uid=None,
    eagerness=0,
    description=None,
):
    """Make migration decorator."""
    if selector is None:
        selector = Selector()

    def wrap(wrappedfunction):
        if description is None:
            assert wrappedfunction.__doc__, 'Missing function docstring!'
            desc = wrappedfunction.__doc__.splitlines()[0]
        else:
            desc = description
        migration = Migration(
            function=wrappedfunction,
            uid=uid,
            description=desc,
            eagerness=eagerness,
        )
        return SelectorMigrationGenerator(migration=migration, selector=selector)

    if function is not None:
        return wrap(function)
    return wrap
