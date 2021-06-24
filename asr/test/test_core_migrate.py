import pytest
import asr
from asr.core.migrate import (
    get_migrations, get_migration_generator, migrate_record
)


@asr.instruction()
def some_instruction():
    """Return 2."""
    return 2


@asr.migration
def a_migration(record):
    """Set a_parameter to 2."""
    record.parameters.a_parameter = 2
    return record


@pytest.mark.ci
def test_migration_is_registered(asr_tmpdir):
    migrations = get_migrations()
    print(migrations)
    assert a_migration in migrations


@pytest.mark.ci
def test_migrate_record(asr_tmpdir):
    make_migrations = get_migration_generator()
    record = some_instruction.get()
    record_migration = migrate_record(record, make_migrations)
    assert record_migration
    assert not record_migration.has_errors()
