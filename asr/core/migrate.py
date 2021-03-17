"""Implements record migration functionality."""
import abc
import typing
from dataclasses import dataclass
from .command import get_recipes
from .selector import Selector
from .record import Record
from .specification import get_new_uuid


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
class MigrationLog:
    """Container for logging migration information."""

    migration_uid: UID
    to_version: UID
    description: str
    modification: Modification

    @classmethod
    def from_migration(
            cls,
            migration: 'Migration',
            modification: Modification,
            to_version: UID,
    ) -> 'MigrationLog':
        return cls(
            migration_uid=migration.uid,
            description=migration.description,
            modification=modification,
            to_version=to_version,
        )


@dataclass
class MigrationHistory:
    """A class the represents the migration history."""

    history: typing.List[MigrationLog]

    def append(self, migration_log: MigrationLog):
        self.history.append(migration_log)

    @property
    def current_version(self) -> typing.Union[None, UID]:
        """Get the current migration version, 'None' if no migrations."""
        if not self.history:
            return None
        return self.history[-1].to_version

    def __contains__(self, migration: 'Migration'):
        return any(migration.uid == log.migration_uid for log in self.history)


@dataclass
class Migration:
    """A class to update a record to a greater version."""

    function: typing.Callable
    uid: UID
    description: str
    eagerness: int = 0

    def apply(self, record: Record) -> Record:
        """Apply migration to record and return mutated record."""
        migrated_record = self.function(record.copy())
        to_version = get_new_uuid()
        mod_cls = DumbModification
        migration_log = MigrationLog.from_migration(
            migration=self,
            modification=mod_cls(
                previous_record=record.copy(),
                new_record=migrated_record.copy(),
            ),
            to_version=to_version,
        )

        if migrated_record.migrations:
            migrated_record.migrations.append(migration_log)
        else:
            migration_history = MigrationHistory(history=[migration_log])
            migrated_record.migrations = migration_history
        return migrated_record

    def __call__(self, record: Record) -> Record:
        return self.apply(record)

    def __str__(self):
        return f'#{self.uid[:5]} "{self.description}"'


class MakeMigrations(abc.ABC):
    """Abstract Base class for factory for making migrations."""

    @abc.abstractmethod
    def make_migrations(self, record: Record) -> typing.List[Migration]:
        pass

    def __call__(self, record: Record) -> typing.List[Migration]:
        return self.make_migrations(record)


@dataclass
class SelectorMigrationGenerator(MakeMigrations):

    selector: Selector
    migration: Migration

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
    applied_migrations: typing.List[Migration]
    errors: typing.List[typing.Tuple[Migration, Exception]]

    def has_migrations(self):
        """Has migrations to apply."""
        return bool(self.applied_migrations)

    def has_errors(self):
        """Has failed migrations."""
        return bool(self.errors)

    def apply(self, cache):
        """Apply record migration to a cache."""
        cache.update(self.migrated_record)

    def __str__(self):
        nmig = len(self.applied_migrations)
        nerr = len(self.errors)
        migrations_string = ' -> '.join([
            str(migration) for migration in self.applied_migrations])
        problem_string = (
            ', '.join(f'{mig} err="{err}"' for mig, err in self.errors)
        )
        return (
            f'UID=#{self.initial_record.uid[:8]} '
            + (f'name={self.initial_record.name}. ')
            + (f'{nmig} migration(s). ' if nmig > 0 else '')
            + (f'{nerr} migration error(s)! ' if self.errors else '')
            + (f'{migrations_string}. ' if nmig > 0 else '')
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
    while True:
        applicable_migrations = migration_generator(migrated_record)
        candidate_migrations = [
            mig for mig in applicable_migrations
            if mig not in problematic_migrations
        ]
        if not candidate_migrations:
            break

        migration = max(candidate_migrations, key=lambda mig: mig.eagerness)
        try:
            migrated_record = migration(migrated_record)
        except Exception as err:
            problematic_migrations.append(migration)
            errors.append((migration, err))
            continue
        applied_migrations.append(migration)
    return RecordMigration(
        initial_record=record,
        migrated_record=migrated_record,
        applied_migrations=applied_migrations,
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
