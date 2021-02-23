"""Implements record migration functionality."""
import typing
from dataclasses import dataclass
from .command import get_recipes
from .selector import Selector
from .record import Record
from .specification import get_new_uuid


class NoMigrationError(Exception):
    """Raise when no migration are needed."""


RecordUID = str


@dataclass
class MigrationLog:
    """Container for logging migration information."""

    migrated_from: typing.Optional[RecordUID]
    migrated_to: typing.Optional[RecordUID]


@dataclass
class RecordMutation:
    """A class to update a record to a greater version."""

    function: typing.Callable
    from_version: int
    to_version: int
    selector: Selector
    description: str

    def apply(self, record: Record) -> Record:
        """Apply mutation to record and return mutated record."""
        assert self.applies_to(record)
        migrated_record = self.function(record.copy())
        migrated_record.uid = get_new_uuid()
        migrated_record.version = self.to_version
        migration_log = MigrationLog(migrated_from=record.uid)
        migrated_record.migration_log = migration_log
        if record.migration_log:
            record.migration_log.migrated_to = migrated_record.uid
        else:
            record.migration_log = MigrationLog(
                migrated_to=migrated_record.uid,
                migrated_from=None,
            )
        return migrated_record

    def applies_to(self, record: Record) -> bool:
        """Check if mutation applies to record."""
        return self.selector.matches(record)

    def __str__(self):
        return f'{self.description} {self.from_version} -> {self.to_version}'


class RecordMigrationFactory:
    """Construct record migrations.

    Manages a collection of RecordMutations and can be told to create
    RecordMigration.
    """

    def __init__(self, mutations):
        self.mutations = []
        for mutation in mutations:
            self.mutations.append(mutation)

    def add(self, mutation: RecordMutation):
        """Add mutation."""
        self.mutations.append(mutation)

    def __call__(self, record: Record) -> typing.Optional['RecordMigration']:
        try:
            sequence_of_mutations = make_migration_strategy(
                self.mutations, record)
        except NoMigrationError:
            return None
        record_migration = RecordMigration(sequence_of_mutations, record)
        return record_migration


def make_migration_strategy(
    mutations: typing.List[RecordMutation],
    record: Record,
) -> typing.List[RecordMutation]:
    """Given mutations and record construct a migration strategy."""
    relevant_mutations = {}
    for mutation in mutations:
        if mutation.applies_to(record):
            relevant_mutations[mutation.from_version] = mutation

    strategy = []
    version = record.version
    if version not in relevant_mutations:
        raise NoMigrationError
    while version in relevant_mutations:
        mutation = relevant_mutations[version]
        strategy.append(mutation)
        version = mutation.to_version
    return strategy


@dataclass
class RecordMigration:
    """A class that represents a record migration."""

    migrations: typing.List[RecordMutation]
    record: Record

    def apply(self, cache):
        """Apply migration to a cache."""
        records = [self.record]
        for migration in self.migrations:
            migrated_record = migration(records[-1])
            records.append(migrated_record)

        original_record, *migrated_records = records
        cache.update(original_record)
        for migrated_record in migrated_records:
            cache.add(migrated_record)


def collect_record_mutations():
    """Collect record mutations from all recipes."""
    recipes = get_recipes()
    mutations = []
    for recipe in recipes:
        if recipe.mutations:
            mutations.extend(recipe.mutations)

    return mutations
