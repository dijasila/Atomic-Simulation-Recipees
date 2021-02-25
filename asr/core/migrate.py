"""Implements record migration functionality."""
import typing
from dataclasses import dataclass
from .command import get_recipes
from .selector import Selector
from .record import Record


class NoMigrationError(Exception):
    """Raise when no migration are needed."""


RecordUID = str


@dataclass
class MutationLog:
    """Container for logging migration information."""

    from_version: int
    to_version: int
    description: str
    previous_record: Record

    @classmethod
    def from_mutation(
            cls, mutation: 'RecordMutation', record: Record) -> 'MutationLog':
        return cls(
            from_version=mutation.from_version,
            to_version=mutation.to_version,
            description=mutation.description,
            previous_record=record,
        )


@dataclass
class MigrationHistory:
    """A class the represents the migration history."""

    history: typing.List[MutationLog]

    def append(self, mutation_log: MutationLog):
        self.history.extend(mutation_log)


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
        migrated_record.version = self.to_version
        mutation_log = MutationLog.from_mutation(self, record)
        if migrated_record.migrations:
            migrated_record.migrations.append(mutation_log)
        else:
            migration_history = MigrationHistory(history=[mutation_log])
            migrated_record.migrations = migration_history
        return migrated_record

    def __call__(self, record: Record) -> Record:
        return self.apply(record)

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

    mutations: typing.List[RecordMutation]
    record: Record

    def apply(self, cache):
        """Apply migration to a cache."""
        migrated_record = self.record
        for mutation in self.mutations:
            migrated_record = mutation(migrated_record)

        cache.update(migrated_record)

    @property
    def from_version(self):
        return self.mutations[0].from_version

    @property
    def to_version(self):
        return self.mutations[-1].to_version

    def __str__(self):
        return (
            f'Migrate record uid={self.record.uid} '
            f'name={self.record.name} '
            f'from version={self.from_version} '
            f'to version={self.to_version}.')


def collect_record_mutations():
    """Collect record mutations from all recipes."""
    recipes = get_recipes()
    mutations = []
    for recipe in recipes:
        if recipe.mutations:
            mutations.extend(recipe.mutations)

    return mutations
