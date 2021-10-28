from pathlib import Path

from ase.db import connect

from asr.core.migrate import records_to_migration_report
from asr.core.resultfile import get_resultfile_records_from_database_row
from asr.core.serialize import JSONSerializer


def collapse_database(databasein: str, databaseout: str):
    assert (
        not databasein == databaseout
    ), "Input and output databases cannot be identical."
    dbin = connect(databasein)
    assert not Path(databaseout).exists()
    with connect(databaseout) as dbout:
        write_collapsed_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def write_collapsed_database(dbin, dbout):
    for ir, row in enumerate(dbin.select("first_class_material=True")):
        if ir % 100 == 0:
            print(ir)
        data = get_data_including_child_data(dbin, row)
        write_row_with_new_data(dbout, row, data)


def write_row_with_new_data(dbout, row, data):
    dbout.write(
            atoms=row.toatoms(),
            key_value_pairs=row.key_value_pairs,
            data=data,
        )


def get_data_including_child_data(dbin, row):
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
        child_row = dbin.get(uid=child_uid)
        children_data[child_uid] = dict(
            directory=child_directory,
            data=child_row.data,
        )

    return children_data


def get_children_from_row(row):
    children = row.data.get("__children__", {})
    return children


def convert_database(databasein: str, databaseout: str):
    assert (
        not databasein == databaseout
    ), "Input and output databases cannot be identical."
    dbin = connect(databasein)
    assert not Path(databaseout).exists()
    with connect(databaseout) as dbout:
        write_converted_database(dbin, dbout)
    copy_database_metadata(dbin, dbout)


def write_converted_database(dbin, dbout):
    serializer = JSONSerializer()
    for row in dbin.select():
        if row.id % 100 == 0:
            print(row.id)
        records = get_resultfile_records_from_database_row(row)
        data = serializer.serialize(dict(records=records))
        write_row_with_new_data(dbout, row, data)


def migrate_database(databasein, databaseout):
    dbin = connect(databasein)
    with connect(databaseout) as dbout:
        write_migrated_database(dbin, dbout)


def write_migrated_database(dbin, dbout):
    ser = JSONSerializer()
    for row in dbin.select():
        print(row.id)
        records = ser.deserialize(ser.serialize(row.data["records"]))
        report = records_to_migration_report(records)
        if report.n_errors == 0 and report.n_applicable_migrations == 0:
            continue
        if report.n_errors > 0:
            report.print_errors()
            break
        data = ser.serialize(dict(records=records))
        write_row_with_new_data(dbout, row, data)
        print(report.summary)
