from pathlib import Path

from asr.database import connect, ASEDatabaseInterface, Row

from asr.core.migrate import records_to_migration_report
from asr.core.resultfile import get_resultfile_records_from_database_row


def collapse_database(databasein: str, databaseout: str):
    assert_databases_not_identical(databasein, databaseout)
    dbin = connect(databasein)
    assert_database_doesnt_exist(databaseout)
    with connect(databaseout) as dbout:
        write_collapsed_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def write_collapsed_database(dbin: ASEDatabaseInterface, dbout: ASEDatabaseInterface):
    for ir, row in enumerate(dbin.select("first_class_material=True")):
        if ir % 100 == 0:
            print(ir)
        assert "records" not in row.data
        data = get_data_including_child_data(dbin, row)
        assert "records" not in data
        write_row_with_new_data(dbout, row, data)


def write_row_with_new_data(
    dbout: ASEDatabaseInterface, row: Row, data=None, records=None
):
    if data is None:
        data = {}
    dbout.write(
        atoms=row.toatoms(),
        key_value_pairs=row.key_value_pairs,
        records=records,
        data=data,
    )


def get_data_including_child_data(dbin: ASEDatabaseInterface, row: Row):
    children = get_children_from_row(row)
    children_data = get_children_data_from_database(dbin, children)
    data = add_children_data(row.data, children_data)
    return data


def copy_database_metadata(dbin, dbout):
    dbout.metadata = dbin.metadata


def add_children_data(data, children_data):
    data = dict(__children_data__=children_data, **data)
    return data


def get_children_data_from_database(dbin, children):
    children_data = {}
    for child_directory, child_uid in children.items():
        try:
            child_row = dbin.get(uid=child_uid)
        except AssertionError:
            # If there are multiple matching child rows
            # we need to use the one that matches child_directory
            child_rows = list(dbin.select(uid=child_uid))
            child_rows = list(
                filter(
                    lambda x: x.folder.endswith(child_directory),
                    child_rows
                )
            )
            assert len(child_rows) == 1, "Cannot find matching child rows."
            child_row = child_rows[0]
        children_data[child_uid] = dict(
            directory=child_directory,
            data=child_row.data,
        )

    return children_data


def get_children_from_row(row: Row):
    children = row.data.get("__children__", {})
    return children


def convert_database(databasein: str, databaseout: str):
    assert_databases_not_identical(databasein, databaseout)
    assert_database_doesnt_exist(databaseout)
    dbin = connect(databasein)
    with connect(databaseout) as dbout:
        write_converted_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def get_other_data_files_from_row(row):
    data = row.data
    other_data = {}
    for name, value in data.items():
        if not name.startswith("results-"):
            other_data[name] = value
    return other_data


def write_converted_database(dbin, dbout):
    for row in dbin.select():
        if row.id % 100 == 0:
            print(row.id)
        records = get_resultfile_records_from_database_row(row)
        data = get_other_data_files_from_row(row)
        assert records
        write_row_with_new_data(dbout, row, data=data, records=records)


def migrate_database(databasein, databaseout):
    assert_databases_not_identical(databasein, databaseout)
    assert_database_doesnt_exist(databaseout)
    dbin = connect(databasein)
    with connect(databaseout) as dbout:
        write_migrated_database(dbin, dbout)


def assert_databases_not_identical(databasein, databaseout):
    assert (
        not databasein == databaseout
    ), "Input and output databases cannot be identical."


def assert_database_doesnt_exist(database):
    assert not Path(database).exists(), f"{database} already exists."


def write_migrated_database(dbin, dbout):
    for row in dbin.select():
        print(row.id)
        records = row.records
        report = records_to_migration_report(records)
        if report.n_errors == 0 and report.n_applicable_migrations == 0:
            continue
        if report.n_errors > 0:
            report.print_errors()
            break
        from asr.core.cache import Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        for record in records:
            cache.add(record)
        for record_migration in report.applicable_migrations:
            record_migration.apply(cache)
        records = cache.select()
        write_row_with_new_data(dbout, row, records=records)
        # print(report.summary)
