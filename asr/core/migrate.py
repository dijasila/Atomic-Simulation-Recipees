"""Implements record mutation functionality."""
import abc
import copy
import os
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .command import get_recipes
from .history import History
from .record import Record
from .selector import Selector
from .specification import get_new_uuid
from .utils import compare_equal


class NonMigratableRecord(Exception):
    """Raise when mutation cannot be used to migrate a Record."""


RecordUID = str
UID = str


@dataclass
class Attribute:
    """Class that represents an object attribute.

    Represents an attribute like ".name" of an object.

    Attributes
    ----------
    name : str
        The name of the attribute.

    """

    name: str

    def set(self, obj: Any, value: Any) -> None:
        setattr(obj, self.name, value)

    def get(self, obj: Any) -> None:
        return getattr(obj, self.name)

    def delete(self, obj: Any) -> None:
        delattr(obj, self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return f".{self.name}"


@dataclass
class Item:
    """Class that represents an object item.

    Represents an item with "name".

    Attributes
    ----------
    name : str
        The name of the item.

    """

    name: str

    def set(self, obj: Any, value: Any) -> None:
        obj[self.name] = value

    def get(self, obj: Any) -> None:
        return obj[self.name]

    def delete(self, obj: Any) -> None:
        del obj[self.name]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Item):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return f'["{self.name}"]'


@dataclass
class ItemAttributeSequence:
    """Class the represents a sequence of attributes and items.

    Class that represents something like obj.attribute[item].attribute2[item2]
    etc.

    Attributes
    ----------
    attrs : List[Union[Attribute, Item]]
        A list of attributes and items to be accessed in that order.

    """

    attrs: List[Union[Attribute, Item]]

    def set(self, obj: Any, value: Any) -> None:
        """Assign value on obj of attr/item specified by sequence."""
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].set(obj, value)

    def get(self, obj: Any) -> Any:
        """Get value on object of attr/item specified by sequence."""
        for attr in self.attrs:
            obj = attr.get(obj)
        return obj

    def delete(self, obj: Any) -> None:
        """Delete attr/item on object specified by sequence."""
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].delete(obj)

    def __getitem__(self, item) -> "ItemAttributeSequence":
        return ItemAttributeSequence(self.attrs[item])

    def __add__(self, other) -> "ItemAttributeSequence":
        return ItemAttributeSequence(self.attrs + other.attrs)

    def __hash__(self) -> int:
        return hash(tuple(hash(attr) for attr in self.attrs))

    def __eq__(self, other: Any) -> bool:
        return self.attrs == other.attrs

    def __str__(self) -> str:
        return "".join(str(attr) for attr in self.attrs)


@dataclass  # type: ignore
class Change(abc.ABC):
    """Abstract class that represent a change of a single attribute/item between records.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was changed.
    """

    attribute: ItemAttributeSequence

    @abc.abstractmethod
    def apply(self, obj: Any) -> None:
        ...

    @abc.abstractmethod
    def revert(self, obj: Any) -> None:
        ...


@dataclass
class NewAttribute(Change):
    """Change object that represents a new attribute.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was changed.
    value : Any
        The new value of the attribute.

    """

    value: Any

    def apply(self, obj: Any) -> None:
        self.attribute.set(obj, copy.deepcopy(self.value))

    def revert(self, obj: Any) -> None:
        self.attribute.delete(obj)

    def __str__(self):
        return f"New attribute={self.attribute} value={self.value}"


@dataclass
class DeletedAttribute(Change):
    """Change object that represents a deleted attribute.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was deleted.
    value : Any
        The old value of the deleted attribute.

    """

    value: Any

    def apply(self, obj: Any):
        self.attribute.delete(obj)

    def revert(self, obj: Any):
        self.attribute.set(obj, copy.deepcopy(self.value))

    def __str__(self):
        return f"Delete attribute={self.attribute} value={self.value}"


@dataclass
class ChangedValue(Change):
    """Change object that represents a change value.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was deleted.
    new_value : Any
        The new value of the attribute.
    old_value : Any
        The old value of the attribute.

    """

    new_value: Any
    old_value: Any

    def apply(self, obj: Any):
        self.attribute.set(obj, copy.deepcopy(self.new_value))

    def revert(self, obj: Any):
        self.attribute.set(obj, copy.deepcopy(self.old_value))

    def __str__(self):
        return (
            f"Change attribute={self.attribute} "
            f"old={self.old_value} new={self.new_value}"
        )


@dataclass
class ChangeCollection:
    """Class that represents multiple differences.

    A modification is basically a collection of differences.

    Attributes
    ----------
    changes : List[Changes]
        A list of difference that together comprises the modification.
    """

    changes: List[Change] = field(default_factory=list)

    def apply(self, record: Record) -> Record:
        """Apply modification to record.

        Parameters
        ----------
        record : Record
            Record to be modified.

        Returns
        -------
        Record
            Modified record.
        """
        for change in self.changes:
            change.apply(record)
        return record

    def revert(self, record: Record) -> Record:
        """Revert modification on record.

        Parameters
        ----------
        record : Record
            Record where modification should be reverted.

        Returns
        -------
        Record
            Reverted record.
        """
        for change in self.changes:
            change.revert(record)
        return record

    def __str__(self):
        return "\n".join(str(diff) for diff in self.changes)

    def __bool__(self):
        return bool(self.changes)


@dataclass
class Revision:
    """Container for logging mutations.

    The revision object is the main building block of the mutation history
    of records. A revision stores a series of mo

    The revision object can be thought of as an analogue to a git commit.
    It has a uid that represents the curre

    """

    uid: UID
    mutation_uid: Optional[UID]
    description: str
    changes: ChangeCollection

    def apply(self, record: Record):
        if record.history is None:
            record.history = RevisionHistory()
        record = self.changes.apply(record.copy())
        record.history.add(self)  # type: ignore
        return record

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            value = str(value)
            if "\n" in value:
                value = "\n" + textwrap.indent(value, " ")
            lines.append(f"{key}={value}")
        return "\n".join(lines)

    def __bool__(self):
        return bool(self.changes)


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

    history: List[Revision] = field(default_factory=list)

    def add(self, revision: Revision):
        """Add revision to history."""
        self.history.append(revision)

    @property
    def latest_revision(self) -> Optional[Revision]:
        """Get the latest revision, 'None' if no revisions."""
        if not self.history:
            return None
        latest_revision = self.history[-1]
        return latest_revision

    def __str__(self):
        lines = [f"latest_revision={self.latest_revision.uid}"]
        for revision in self.history:
            value = str(revision)
            if "\n" in value:
                value = "\n" + textwrap.indent(value, " ")
            lines.append(f"revision={value}")
        return "\n".join(lines)


@dataclass
class Mutation:
    """A class to update a record to a greater version."""

    function: Callable
    description: str
    uid: Optional[UID] = None
    eagerness: int = 0
    selector: Callable[[Record], bool] = field(default_factory=Selector)

    def applies(self, record: Record) -> bool:
        return self.selector(record)

    def apply(self, record: Record) -> Revision:
        """Apply mutation to record and return record Revision."""
        mutated_record = self.function(record.copy())
        changes = make_change_collection(
            record,
            mutated_record,
        )
        revision = Revision(
            description=self.description,
            changes=changes,
            mutation_uid=self.uid,
            uid=get_new_uuid(),
        )

        return revision

    def __call__(self, record: Record) -> Revision:
        return self.apply(record)

    def __str__(self):
        return self.description

    def __hash__(self):
        return hash(id(self))


@dataclass
class MutationCollection:
    """Generates the applicable mutations from a collection of mutations."""

    mutations: Set[Mutation] = field(default_factory=set)

    def add(self, mutations: List[Mutation]) -> None:
        self.mutations.update(mutations)

    def get_applicable_mutations(self, record: Record) -> List[Mutation]:
        applicable_mutations = [
            mutation for mutation in self.mutations if mutation.applies(record)
        ]
        return applicable_mutations

    def __contains__(self, mutation: Mutation) -> bool:
        return mutation in self.mutations


@dataclass
class Migration:
    """A class that represents a record migration."""

    initial_record: Record
    migrated_record: Record
    revisions: List[Revision]
    errors: List[Tuple[Mutation, Exception]]

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
            f"record.uid={self.initial_record.uid}",
        ]
        if nrev:
            revisions_string = "\n".join(
                [
                    f"Revision #{i} {revision}"
                    for i, revision in enumerate(self.revisions)
                ]
            )
            items.append(revisions_string)
        if nerr:
            problem_string = ", ".join(f'{mig} err="{err}"' for mig, err in self.errors)
            items.append(problem_string)
        return "\n".join(items)


@dataclass
class MigrationReport:
    applicable_migrations: List[Migration]
    erroneous_migrations: List[Migration]
    n_up_to_date: int
    n_applicable_migrations: int
    n_errors: int

    @property
    def summary(self) -> str:
        return "\n".join(
            [
                f"There are {self.n_applicable_migrations} unapplied migrations, "
                f"{self.n_errors} erroneous migrations and "
                f"{self.n_up_to_date} records are up to date.",
                "",
            ]
        )

    def print_errors(self) -> None:
        for record_migration in self.erroneous_migrations:
            print(f"Error for: {record_migration}")
            for migration, error in record_migration.errors:
                print(f"Error in: {migration}")
                traceback.print_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
                print()

    @property
    def verbose(self) -> str:
        strs = []
        for i, migration in enumerate(self.applicable_migrations):
            strs.append(f"#{i} {migration}")
        return "\n\n".join(strs)


def make_change_collection(old_record: Record, new_record: Record) -> ChangeCollection:
    """Search for changes between objects and make resulting modification."""
    changes = get_changes(old_record, new_record)
    return ChangeCollection(changes)


def get_changes(
    obj1: Any, obj2: Any, prepend: Optional[ItemAttributeSequence] = None
) -> List[Change]:
    """Get differences from obj1 to obj2.

    Parameters
    ----------
    obj1 : Any
        An object before changes has been made.
    obj2 : Any
        An object after changes has been made.
    prepend : Optional[ItemAttributeSequence], optional
        Prepend item/attribute sequences by this, by default None. This is used,
        internally by the algorithm when recursing into the objects be introspected.
        Usually you don't need this.

    Returns
    -------
    List[Difference]
        List of difference objects that represents the difference from obj1 to obj2.
    """
    if prepend is None:
        prepend = ItemAttributeSequence([])
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
            assert prepend.attrs[0].name != "history"
            return [
                ChangedValue(
                    attribute=prepend,
                    old_value=copy.deepcopy(obj1),
                    new_value=copy.deepcopy(obj2),
                )
            ]
    differences: List[Change] = []
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
        diffs_inside_values = get_changes(value1, value2, prepend=prepend + attr)
        differences.extend(diffs_inside_values)

    return differences


def get_attributes_and_values(
    obj: Any,
) -> Dict[ItemAttributeSequence, Any]:
    """Get dict of attributes and values of obj.

    Parameters
    ----------
    obj : Any
        Object to be introspected

    Returns
    -------
    Dict[ItemAttributeSequence, Any]
        Dictionary that maps attributes to values.
    """
    attributes_and_values = {}
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            attributes_and_values[ItemAttributeSequence([Attribute(key)])] = value
    elif hasattr(obj, "__slots__"):
        for key in obj.__slots__:
            value = getattr(obj, key)
            attributes_and_values[ItemAttributeSequence([Attribute(key)])] = value
    elif isinstance(obj, dict):
        for key, value in obj.items():
            attr = ItemAttributeSequence([Item(key)])
            attributes_and_values[attr] = value

    return attributes_and_values


def migrate_record(
    record: Record,
    mutations: MutationCollection,
) -> Migration:
    """Construct a record migration."""
    migrated_record = record.copy()
    applied_mutations = []
    problematic_mutations = []
    errors: List[Tuple[Mutation, Exception]] = []
    revisions = []
    while True:
        applicable_mutations = mutations.get_applicable_mutations(migrated_record)
        candidate_mutations = [
            mut
            for mut in applicable_mutations
            if (mut not in problematic_mutations and mut not in applied_mutations)
        ]
        if not candidate_mutations:
            break

        candidate_mutation = max(candidate_mutations, key=lambda mig: mig.eagerness)
        try:
            revision = candidate_mutation(migrated_record)
        except NonMigratableRecord as err:
            problematic_mutations.append(candidate_mutation)
            errors.append((candidate_mutation, err))
            continue
        except Exception as err:  # pylint: disable=broad-except
            problematic_mutations.append(candidate_mutation)
            errors.append((candidate_mutation, err))
            if os.environ.get("ASR_DEBUG", False):
                raise
            continue
        applied_mutations.append(candidate_mutation)
        if not revision:
            continue
        migrated_record = revision.apply(migrated_record)
        revisions.append(revision)
    return Migration(
        initial_record=record,
        revisions=revisions,
        migrated_record=migrated_record,
        errors=errors,
    )


def mutation(
    function=None,
    *,
    selector=None,
    uid=None,
    eagerness=0,
    description=None,
):
    """Mutation decorator.

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
        (optional) Description of the mutation. Default is to use the
        docstring of the decorated function.
    """
    if selector is None:
        selector = Selector()

    def wrap(wrappedfunction):
        if description is None:
            assert wrappedfunction.__doc__, "Missing function docstring!"
            desc = wrappedfunction.__doc__.splitlines()[0]
        else:
            desc = description
        mut = Mutation(
            function=wrappedfunction,
            uid=uid,
            description=desc,
            eagerness=eagerness,
            selector=selector,
        )
        register_mutation(mut)
        return mut

    if function is not None:
        mut = wrap(function)
    else:
        mut = wrap
    return mut


MUTATIONS: MutationCollection = MutationCollection()


def register_mutation(mutation_to_be_registered: Mutation) -> None:
    """Register a mutation.

    Parameters
    ----------
    mutation : Mutation
        Mutation to be registered.
    """
    MUTATIONS.add([mutation_to_be_registered])


def get_mutations() -> MutationCollection:
    """Get registered migrations."""
    # We import all recipes to make sure the Mutations have been registered.
    get_recipes()
    return MUTATIONS


def make_migrations(
    records: List[Record],
    mutations: MutationCollection,
) -> List[Migration]:
    migrations = []
    for record in records:
        migration = migrate_record(record, mutations)
        migrations.append(migration)
    return migrations


def make_migration_report(migrations: List[Migration]) -> MigrationReport:
    erroneous_migrations = []
    n_up_to_date = 0
    n_applicable_migrations = 0
    n_errors = 0
    applicable_migrations = []
    for migration in migrations:
        if migration:
            n_applicable_migrations += 1
            applicable_migrations.append(migration)

        if migration.has_errors():
            n_errors += 1
            erroneous_migrations.append(migration)

        if not (migration or migration.has_errors()):
            n_up_to_date += 1

    return MigrationReport(
        applicable_migrations=applicable_migrations,
        erroneous_migrations=erroneous_migrations,
        n_up_to_date=n_up_to_date,
        n_applicable_migrations=n_applicable_migrations,
        n_errors=n_errors,
    )


def records_to_migration_report(records: List[Record]) -> MigrationReport:
    mutations = get_mutations()
    record_migrations = make_migrations(records, mutations)
    report = make_migration_report(record_migrations)
    return report
