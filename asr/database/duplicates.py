from asr.core import command, argument, option
from asr.duplicates import check_duplicates
from functools import cmp_to_key
from datetime import datetime


@command(module='asr.database.duplicates',
         resources='1:20m',
         save_results_file=False)
@argument('databaseout')
@argument('database')
@option('-f', '--filterstring',
        help='Comparison string. If true then pick row1, else pick row2.')
def main(database, databaseout,
         filterstring='natoms,id'):
    """Take an input database filter out duplicates.

    Uses asr.duplicates.check_duplicates.

    """
    from ase.db import connect
    assert database != databaseout, \
        'You cannot read and write from the same database.'
    db = connect(database)
    already_checked_set = set()
    nmat = len(db)
    with connect(databaseout) as filtereddb:
        for row in db.select(include_data=False):
            _timed_print(f'{row.id}/{nmat}', wait=30)
            if row.id % 200 == 0:
                now = datetime.now()
                print(f'{now}: {row.id}')

            if row.id in already_checked_set:
                continue

            structure = row.toatoms()
            ref_mag = row.get('magstate')
            already_checked_set.add(row.id)
            has_duplicate, id_list = check_duplicates(
                structure, db, ref_mag,
                exclude_ids=already_checked_set)
            already_checked_set.update(set(id_list))

            if has_duplicate:
                print(f'row.id={row.id}: {row.formula} has '
                      f'duplicates with ids: {id_list}')
                duplicate_ids = set(id_list + [row.id])
                relevant_row = pick_out_row(db, duplicate_ids, filterstring)
            else:
                relevant_row = row

            filtereddb.write(atoms=relevant_row.toatoms(),
                             data=relevant_row.data,
                             **relevant_row.key_value_pairs)

    filtereddb.metadata = db.metadata


_LATEST_PRINT = datetime.now()


def _timed_print(*args, wait=20):
    global _LATEST_PRINT

    now = datetime.now()
    if (now - _LATEST_PRINT).seconds > wait:
        print(*args)
        _LATEST_PRINT = now


def pick_out_row(db, duplicate_ids, filterstring):
    rows = [db.get(id=rowid) for rowid in duplicate_ids]
    keys = filterstring.split(',')

    def keyfunc(row):
        return tuple(row.get(key) for key in keys)

    rows = sorted(rows, key=keyfunc)
    return rows[0]


if __name__ == '__main__':
    main.cli()
