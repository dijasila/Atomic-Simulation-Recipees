import os
from pathlib import Path

import pytest

from asr.database import DatabaseProject, connect
from asr.database.migrate import (
    write_collapsed_database,
    write_converted_database,
    write_migrated_database,
)

from .fixtures import get_app_row_contents


ASR_TEST_DIR = os.environ.get("ASR_TEST_DATA")

if ASR_TEST_DIR is not None:
    DIRECTORY_WITH_DATABASES_TO_BE_MIGRATED = (
        ASR_TEST_DIR + "/ASR_MIGRATION_TEST_DATABASES"
    )
    databases_to_be_migrated = [
        (False, pth)
        for pth in Path(DIRECTORY_WITH_DATABASES_TO_BE_MIGRATED).glob("*.db")
    ]
    DIRECTORY_WITH_DATABASES_TO_BE_COLLAPSED = (
        ASR_TEST_DIR + "/ASR_COLLAPSE_TEST_DATABASES"
    )
    databases_to_be_collapsed = [
        (True, pth)
        for pth in Path(DIRECTORY_WITH_DATABASES_TO_BE_COLLAPSED).glob("*.db")
    ]
    databases_to_be_migrated.extend(databases_to_be_collapsed)
else:
    databases_to_be_migrated = []


@pytest.fixture(
    params=databases_to_be_migrated,
    ids=[str(pth[1].absolute()) for pth in databases_to_be_migrated],
)
def database_to_be_migrated(request):
    collapse, database_name = request.param
    db = connect(database_name)
    return collapse, db


@pytest.mark.skipif(
    not databases_to_be_migrated,
    reason="No databases to test migration on.",
)
@pytest.mark.ci
def test_collapse_database(database_to_be_migrated, asr_tmpdir):
    collapse, database_to_be_migrated = database_to_be_migrated
    if "defect" in database_to_be_migrated.db.filename:
        pytest.xfail("Defect databases cannot be migrated yet.")
    original_metadata = database_to_be_migrated.metadata
    if collapse:
        original_row = list(database_to_be_migrated.select(first_class_material=True))[
            0
        ]
    else:
        original_row = database_to_be_migrated.get(id=1)
    original_row_extra_data_files = get_extra_data_files(original_row)
    children_key = set(["__children__"])
    children_data_key = set(["__children_data__"])
    record_key = set(["records"])
    if collapse:
        with connect("collapsed.db") as collapsed:
            write_collapsed_database(database_to_be_migrated, collapsed)
        row = collapsed.get(id=1)
        assert original_metadata == collapsed.metadata
        assert (
            original_row_extra_data_files | children_data_key - children_key
            == get_extra_data_files(row)
        )
        assert "records" not in row.data
        assert "records" not in row.row.data
        database_to_be_migrated = collapsed

    with connect("converted.db") as converted:
        write_converted_database(database_to_be_migrated, converted)
    row = converted.get(id=1)
    assert "records" in row.data
    assert (
        original_row_extra_data_files - children_key | record_key
        == get_extra_data_files(row)
    )

    records = row.records
    nrecords = len(records)
    assert nrecords > 0
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
    row = migrated.get(id=1)
    assert "records" in row.data
    assert (
        original_row_extra_data_files - children_key | record_key
        == get_extra_data_files(row)
    )
    assert nrecords == len(row.records)

    assert original_metadata == migrated.metadata
    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    project = DatabaseProject(
        name="migrated_database",
        title="Migrated database",
        database=migrated,
        tmpdir=tmpdir,
    )
    get_app_row_contents(project)


def get_extra_data_files(original_row):
    return set(
        key for key in original_row.data.keys() if not key.startswith("results-")
    )
