"""Implements record migration functionality."""
import typing
from dataclasses import dataclass
from .command import get_recipes
from .selector import Selector
from .record import Record


class NoMigrationError(Exception):
    """Raise when no migration are needed."""


RecordUID = str
UID = str


@dataclass
class MutationLog:
    """Container for logging migration information."""

    uid: UID
    description: str
    previous_record: Record

    @classmethod
    def from_mutation(
            cls, mutation: 'RecordMutation', record: Record) -> 'MutationLog':
        return cls(
            uid=mutation.uid,
            description=mutation.description,
            previous_record=record,
        )


@dataclass
class MigrationHistory:
    """A class the represents the migration history."""

    history: typing.List[MutationLog]

    def append(self, mutation_log: MutationLog):
        self.history.extend(mutation_log)

    def __contains__(self, mutation: 'RecordMutation'):
        return any(mutation.uid == tmp.uid for tmp in self.history)


@dataclass
class RecordMutation:
    """A class to update a record to a greater version."""

    function: typing.Callable
    selector: Selector
    uid: UID
    description: str

    def apply(self, record: Record) -> Record:
        """Apply mutation to record and return mutated record."""
        assert self.applies(record)
        migrated_record = self.function(record.copy())
        mutation_log = MutationLog.from_mutation(self, record)
        if migrated_record.migrations:
            migrated_record.migrations.append(mutation_log)
        else:
            migration_history = MigrationHistory(history=[mutation_log])
            migrated_record.migrations = migration_history
        return migrated_record

    def __call__(self, record: Record) -> Record:
        return self.apply(record)

    def applies(self, record: Record) -> bool:
        """Check if mutation applies to record."""
        is_match = self.selector.matches(record)
        if record.migrations:
            return (
                is_match
                and self not in record.migrations
            )
        else:
            return is_match

    def __str__(self):
        return f'[uid={self.uid[:5]}...]{self.description}'


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

    record: Record
    mutations: typing.List[RecordMutation]

    def __bool__(self):
        if not self.mutations:
            return False

        does_any_mutation_apply = any(
            mutation.applies(self.record)
            for mutation in self.mutations
        )
        return does_any_mutation_apply

    def run(self):
        assert self

        migrated_record = self.record
        applied_mutations = []
        while True:
            applicable_mutations = [
                mutation
                for mutation in self.mutations
                if (
                    mutation.applies(migrated_record)
                )
            ]
            if not applicable_mutations:
                break
            mutation = applicable_mutations[0]
            migrated_record = mutation(migrated_record)
            applied_mutations.append(mutation)
        return migrated_record, applied_mutations

    def apply(self, cache):
        """Apply migration to a cache."""
        migrated_record, applied_mutations = self.run()
        cache.update(migrated_record)

    def __str__(self):
        _, applied_mutations = self.run()
        mutations_string = ' -> '.join([
            str(mutation) for mutation in applied_mutations])
        return (
            f'Migrate record uid={self.record.uid[:8]}... '
            f'name={self.record.name} '
            f'{mutations_string}'
        )


def collect_record_mutations():
    """Collect record mutations from all recipes."""
    recipes = get_recipes()
    mutations = []
    for recipe in recipes:
        if recipe.mutations:
            mutations.extend(recipe.mutations)

    return mutations
