import os
from pathlib import Path

import pytest

from asr.database import connect
from asr.database.migrate import (
    write_collapsed_database,
    write_converted_database,
    write_migrated_database,
)

ASR_TEST_DIR = os.environ.get("ASR_TEST_DATA")

if ASR_TEST_DIR is not None:
    DIRECTORY_WITH_DATABASES_TO_BE_MIGRATED = (
        ASR_TEST_DIR
        + "/ASR_MIGRATION_TEST_DATABASES"
    )
    databases_to_be_migrated = list(
        Path(DIRECTORY_WITH_DATABASES_TO_BE_MIGRATED).glob("*.db")
    )
else:
    databases_to_be_migrated = []


@pytest.fixture(
    params=databases_to_be_migrated,
    ids=[str(pth.absolute()) for pth in databases_to_be_migrated],
)
def database_to_be_migrated(request):
    database_name = request.param
    db = connect(database_name)
    return db


@pytest.mark.skipif(
    not databases_to_be_migrated,
    reason="No databases to test migration on.",
)
@pytest.mark.ci
def test_collapse_database(database_to_be_migrated, asr_tmpdir):
    with connect("collapsed.db") as collapsed:
        write_collapsed_database(database_to_be_migrated, collapsed)
    row = collapsed.get(id=1)
    assert "records" not in row.data
    assert "records" not in row.row.data

    with connect("converted.db") as converted:
        write_converted_database(collapsed, converted)
    row = converted.get(id=1)
    assert "records" in row.data

    records = row.records
    gwrecords = [rec for rec in records if rec.name == "asr.c2db.gw:main"]
    if gwrecords:
        assert len(gwrecords) == 1
        gwrecord = gwrecords[0]
        assert (
            "asr.c2db.bandstructure:calculate"
            in gwrecord.parameters.dependency_parameters
        )

    with connect("migrated.db") as migrated:
        write_migrated_database(converted, migrated)
    assert "records" in migrated.get(id=1).data
