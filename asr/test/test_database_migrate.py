from pathlib import Path

import pytest
from ase.db import connect
from asr.core.resultfile import add_main_to_name_if_missing

from asr.database.migrate import (
    write_converted_database,
    write_collapsed_database,
    write_migrated_database,
)


@pytest.fixture
def database_to_be_collapsed():
    db = connect(Path(__file__).parent / "data/mos2_collapsable_database.db")
    return db


@pytest.mark.ci
def test_collapse_database(database_to_be_collapsed, asr_tmpdir):
    assert len(database_to_be_collapsed) == 31
    collapsed = connect("collapsed.db")
    write_collapsed_database(database_to_be_collapsed, collapsed)
    assert len(collapsed) == 1

    converted = connect("converted.db")
    write_converted_database(collapsed, converted)
    row = converted.get(id=1)
    assert "records" in row.data

    records = row.data['records']
    gwrecord = [rec.name for rec in records if rec.name == "asr.c2db.gw:main"]

    
    migrated = connect("migrated.db")
    write_migrated_database(converted, migrated)
    assert "records" in migrated.get(id=1).data
