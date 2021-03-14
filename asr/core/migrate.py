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
    """Class that represents a record modification."""

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
            record: Record,
            version: UID,
    ) -> 'MigrationLog':
        return cls(
            migration_uid=migration.uid,
            description=migration.description,
            modification=record,
            to_version=version,
        )


@dataclass
class MigrationHistory:
    """A class the represents the migration history."""

    history: typing.List[MigrationLog]

    def append(self, migration_log: MigrationLog):
        self.history.extend(migration_log)

    @property
    def current_version(self):
        if not self.history:
            return None
        return self.history[-1].to_version

    def __contains__(self, migration: 'Migration'):
        return any(migration.uid == tmp.uid for tmp in self.history)


@dataclass
class Migration:
    """A class to update a record to a greater version."""

    function: typing.Callable
    uid: UID
    description: str

    def apply(self, record: Record) -> Record:
        """Apply migration to record and return mutated record."""
        migrated_record = self.function(record.copy())
        migrated_version = get_new_uuid()
        migration_log = MigrationLog.from_migration(
            self, record, version=migrated_version)

        if migrated_record.migrations:
            migrated_record.migrations.append(migration_log)
        else:
            migration_history = MigrationHistory(history=[migration_log])
            migrated_record.migrations = migration_history
        return migrated_record

    def __call__(self, record: Record) -> Record:
        return self.apply(record)

    def __str__(self):
        return f'#{self.uid[:5]} {self.description}'


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
class CollectionMigrationGenerator(MakeMigrations):

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

    record: Record
    migration_generator: MakeMigrations

    def __bool__(self):
        does_any_migrations_exist = bool(self.migration_generator(self.record))
        return does_any_migrations_exist

    def run(self):
        assert self

        migrated_record = self.record
        applied_migrations = []
        while True:
            applicable_migrations = self.migration_generator(migrated_record)
            if not applicable_migrations:
                break
            for migration in applicable_migrations:
                try:
                    migrated_record = migration(migrated_record)
                except UnapplicableMigration:
                    continue
                applied_migrations.append(migration)
        return migrated_record, applied_migrations

    def apply(self, cache):
        """Apply migration to a cache."""
        migrated_record, _ = self.run()
        cache.update(migrated_record)

    def __str__(self):
        _, applied_migrations = self.run()
        migrations_string = ' -> '.join([
            str(migrations) for migrations in applied_migrations])
        return (
            f'Migrate record #{self.record.uid[:8]} '
            f'name={self.record.name}: '
            f'{migrations_string}'
        )


def get_instruction_migration_generator() -> CollectionMigrationGenerator:
    """Collect record migrations from all recipes."""
    recipes = get_recipes()
    migrations = CollectionMigrationGenerator(migration_generators=[])
    for recipe in recipes:
        if recipe.migrations:
            migrations.extend(recipe.migrations)

    return migrations
