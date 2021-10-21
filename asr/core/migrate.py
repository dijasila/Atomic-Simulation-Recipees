"""Implements record migration functionality."""
import textwrap
import abc
import copy
import typing
import traceback
from dataclasses import dataclass, field
from .command import get_recipes
from .selector import Selector
from .record import Record
from .specification import get_new_uuid
from .history import History
from .utils import compare_equal


class UnapplicableMigration(Exception):
    """Raise when migration doesn't apply."""


RecordUID = str
UID = str


@dataclass
class Modification:
    """Class that represents a record modification."""

    differences: typing.List['Difference'] = field(default_factory=list)

    def apply(self, record: Record) -> Record:
        for difference in self.differences:
            difference.apply(record)
        return record

    def revert(self, record: Record) -> Record:
        for difference in self.differences:
            difference.revert(record)
        return record

    def __str__(self):
        return '\n'.join(str(diff) for diff in self.differences)


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
        self.attribute.set(obj, copy.deepcopy(self.value))

    def revert(self, obj: typing.Any):
        self.attribute.delete(obj)

    def __str__(self):
        return f'New attribute={self.attribute} value={self.value}'


@dataclass
class DeletedAttribute(Difference):

    value: typing.Any

    def apply(self, obj: typing.Any):
        self.attribute.delete(obj)

    def revert(self, obj: typing.Any):
        self.attribute.set(obj, copy.deepcopy(self.value))

    def __str__(self):
        return f'Delete attribute={self.attribute} value={self.value}'


@dataclass
class ChangedValue(Difference):
    new_value: typing.Any
    old_value: typing.Any

    def apply(self, obj: typing.Any):
        self.attribute.set(obj, copy.deepcopy(self.new_value))

    def revert(self, obj: typing.Any):
        self.attribute.set(obj, copy.deepcopy(self.old_value))

    def __str__(self):
        return (
            f'Change attribute={self.attribute} '
            f'old={self.old_value} new={self.new_value}'
        )


@dataclass
class Attribute:
    """Class that represents an object attribute."""

    name: str

    def set(self, obj, value):
        setattr(obj, self.name, value)

    def get(self, obj):
        return getattr(obj, self.name)

    def delete(self, obj):
        delattr(obj, self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name

    def __str__(self):
        return f'.{self.name}'


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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return self.name == other.name

    def __str__(self):
        return f'["{self.name}"]'


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

    def __hash__(self):
        return hash(tuple(hash(attr) for attr in self.attrs))

    def __eq__(self, other):
        return self.attrs == other.attrs

    def __str__(self):
        return ''.join(str(attr) for attr in self.attrs)


def make_modification(old_record: Record, new_record: Record):
    """Search for differences between objects and make resulting modification."""
    differences = get_differences(old_record, new_record)
    return Modification(differences)


def get_differences(obj1, obj2, prepend: typing.Optional[AttributeSequence] = None):
    if prepend is None:
        prepend = AttributeSequence([])
    tp1 = type(obj1)
    tp2 = type(obj2)
    if tp1 != tp2:
        return [
            ChangedValue(
                attribute=prepend,
                old_value=copy.deepcopy(obj1),
                new_value=copy.deepcopy(obj2),
            )
        ]
    attrs_and_values1 = get_attributes_and_values(obj1)
    attrs_and_values2 = get_attributes_and_values(obj2)
    if not (attrs_and_values1 or attrs_and_values2):
        # Then we cannot introspect
        if not compare_equal(obj1, obj2):
            assert prepend.attrs[0].name != 'history'
            return [
                ChangedValue(
                    attribute=prepend,
                    old_value=copy.deepcopy(obj1),
                    new_value=copy.deepcopy(obj2),
                )
            ]
    differences = []
    attrs1 = set(attrs_and_values1)
    attrs2 = set(attrs_and_values2)
    deleted_attrs = attrs1 - attrs2
    new_attrs = attrs2 - attrs1

    for attr in deleted_attrs:
        differences.append(
            DeletedAttribute(
                attribute=prepend + attr,
                value=copy.deepcopy(attrs_and_values1[attr]),
            )
        )

    for attr in new_attrs:
        differences.append(
            NewAttribute(
                attribute=prepend + attr,
                value=copy.deepcopy(attrs_and_values2[attr]),
            )
        )
    common_attrs = attrs1.intersection(attrs2)
    for attr in common_attrs:
        value1 = attrs_and_values1[attr]
        value2 = attrs_and_values2[attr]
        diffs_inside_values = get_differences(value1, value2, prepend=prepend + attr)
        differences.extend(diffs_inside_values)

    return differences


def get_attributes_and_values(obj):
    attributes_and_values = {}
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            attributes_and_values[AttributeSequence([Attribute(key)])] = value
    elif hasattr(obj, '__slots__'):
        for key in obj.__slots__:
            value = getattr(obj, key)
            attributes_and_values[AttributeSequence([Attribute(key)])] = value
    elif isinstance(obj, dict):
        for key, value in obj.items():
            attr = AttributeSequence([Item(key)])
            attributes_and_values[attr] = value

    return attributes_and_values


@dataclass
class Revision:
    """Container for logging migration information."""

    migration_uid: typing.Optional[UID]
    uid: UID
    description: str
    modification: Modification

    def apply(self, record: Record):
        if record.history is None:
            record.history = RevisionHistory()
        record = self.modification.apply(record.copy())
        record.history.add(self)
        return record

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            value = str(value)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'{key}={value}')
        return '\n'.join(lines)


@dataclass
class RevisionHistory(History):
    """A class that represents the revision history.

    Attributes
    ----------
    history
        A chronological list of the revisions that led to the latest
        (current) revision. The latest revision is the last element of this
        list.
    """

    history: typing.List[Revision] = field(default_factory=list)

    def add(self, revision: Revision):
        """Add revision to history."""
        self.history.append(revision)

    @property
    def latest_revision(self) -> typing.Union[None, UID]:
        """Get the latest revision, 'None' if no revisions."""
        if not self.history:
            return None
        latest_revision = self.history[-1]
        return latest_revision

    def __str__(self):
        lines = [f'latest_revision={self.latest_revision.uid}']
        for revision in self.history:
            value = str(revision)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'revision={value}')
        return '\n'.join(lines)


@dataclass
class Migration:
    """A class to update a record to a greater version."""

    function: typing.Callable
    description: str
    uid: typing.Optional[UID] = None
    eagerness: int = 0

    def apply(self, record: Record) -> Revision:
        """Apply migration to record and return record Revision."""
        migrated_record = self.function(record.copy())
        modification = make_modification(
            record,
            migrated_record,
        )
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
        is_match = self.selector(record)
        if is_match:
            return [self.migration]
        else:
            return []

    def __hash__(self):
        return hash(id(self))


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
        self.migration_generators = (
            self.migration_generators
            + list(migration_generators)
        )

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

        items = [
            f'record.uid={self.initial_record.uid}',
        ]
        if nrev:
            revisions_string = '\n'.join([
                f'Revision #{i} {revision}'
                for i, revision in enumerate(self.revisions)])
            items.append(revisions_string)
        if nerr:
            problem_string = (
                ', '.join(f'{mig} err="{err}"' for mig, err in self.errors)
            )
            items.append(problem_string)
        return '\n'.join(items)


def migrate_record(
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
        revisions.append(revision)
    return RecordMigration(
        initial_record=record,
        revisions=revisions,
        migrated_record=migrated_record,
        errors=errors,
    )


def get_instruction_migration_generator() -> CollectionMigrationGenerator:
    """Collect record migrations from all recipes."""
    # Import all recipes to make sure that migrations are registered
    get_recipes()
    migrations = CollectionMigrationGenerator(migration_generators=[])
    migrations.extend(get_migrations())
    return migrations


def get_custom_migrations_generator() -> CollectionMigrationGenerator:
    from .migrations import custom_migrations
    make_migrations = CollectionMigrationGenerator(
        migration_generators=custom_migrations,
    )
    return make_migrations


def get_migration_generator() -> CollectionMigrationGenerator:
    """Return a migration generator that yields all migrations."""
    from asr.core.resultfile import get_resultfile_migration_generator
    make_migrations = get_instruction_migration_generator()
    make_migrations.extend([get_resultfile_migration_generator()])
    make_migrations.extend([get_custom_migrations_generator()])
    return make_migrations


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
    """Migration decorator.

    Parameters
    ----------
    selector
        Callable that returns a boolean used to select records to be migrated.
        Will be applied to all records in the cache.
    uid
        (optional) :func:`uuid.uuid4` uid which can be used to identify migration.
    eagerness
        Integer representing how eager the migration is to be applied. Migrations
        with higher eagerness will take priority over other migrations with lower
        values. Default is 0.
    description
        (optional) Description of the migration. Default is to use the
        docstring of the decorated function.
    """
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
        mig = SelectorMigrationGenerator(migration=migration, selector=selector)
        register_migration(mig)
        return mig

    if function is not None:
        migration = wrap(function)
    else:
        migration = wrap
    return migration


MIGRATIONS = set()


def register_migration(migration) -> None:
    MIGRATIONS.add(migration)


def get_migrations() -> set:
    return MIGRATIONS


def records_to_migration_report(records):
    record_migrations = make_record_migrations(records)
    report = make_migration_report(record_migrations)
    return report


def make_record_migrations(records, make_migrations=None):
    from asr.core.migrate import get_migration_generator, migrate_record
    if make_migrations is None:
        make_migrations = get_migration_generator()
    record_migrations = []
    for record in records:
        record_migration = migrate_record(record, make_migrations)
        record_migrations.append(record_migration)
    return record_migrations


def make_migration_report(record_migrations):
    erroneous_migrations = []
    n_up_to_date = 0
    n_applicable_migrations = 0
    n_errors = 0
    applicable_migrations = []
    for record_migration in record_migrations:
        if record_migration:
            n_applicable_migrations += 1
            applicable_migrations.append(record_migration)

        if record_migration.has_errors():
            n_errors += 1
            erroneous_migrations.append(record_migration)

        if not (record_migration
                or record_migration.has_errors()):
            n_up_to_date += 1

    return MigrationReport(
        applicable_migrations=applicable_migrations,
        erroneous_migrations=erroneous_migrations,
        n_up_to_date=n_up_to_date,
        n_applicable_migrations=n_applicable_migrations,
        n_errors=n_errors,
    )


@dataclass
class MigrationReport:
    applicable_migrations: typing.List[RecordMigration]
    erroneous_migrations: typing.List[RecordMigration]
    n_up_to_date: int
    n_applicable_migrations: int
    n_errors: int

    @property
    def summary(self):
        return '\n'.join(
            [
                f'There are {self.n_applicable_migrations} unapplied migrations, '
                f'{self.n_errors} erroneous migrations and '
                f'{self.n_up_to_date} records are up to date.',
                '',
            ]
        )

    def print_errors(self):
        for record_migration in self.erroneous_migrations:
            print(f'Error for: {record_migration}')
            for migration, error in record_migration.errors:
                print(f'Error in: {migration}')
                traceback.print_exception(
                    type(error), error, error.__traceback__,
                )
                print()

    @property
    def verbose(self):
        strs = []
        for i, migration in enumerate(self.applicable_migrations):
            strs.append(f'#{i} {migration}')
        return '\n\n'.join(strs)
