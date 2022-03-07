import pytest
import asr
from asr.core.migrate import (
    get_mutations, migrate_record
)


@asr.instruction()
def some_instruction():
    """Return 2."""
    return 2


sel = asr.Selector()
sel.name = sel.EQ("asr.test.test_core_migrate:some_instruction")


@asr.mutation(selector=sel)
def a_mutation(record):
    """Set a_parameter to 2."""
    record.parameters.a_parameter = 2
    return record


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_mutation_is_registered(asr_tmpdir):
    mutations = get_mutations()
    print(mutations)
    assert a_mutation in mutations


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_migrate_record(asr_tmpdir):
    mutations = get_mutations()
    record = some_instruction.get()
    migration = migrate_record(record, mutations)
    assert migration
    assert not migration.has_errors()
