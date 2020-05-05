from asr.core import command, argument, option
from asr.duplicates import check_duplicates
from functools import cmp_to_key


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
    from datetime import datetime
    assert database != databaseout, \
        'You cannot read and write from the same database.'
    db = connect(database)
    already_checked_set = set()
    with connect(databaseout) as filtereddb:
        for row in db.select():
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


def pick_out_row(db, duplicate_ids, filterstring):
    rows = [db.get(id=rowid) for rowid in duplicate_ids]
    filters = filterstring.split(',')

    def compare(row1, row2):
        for filt in filters:
            cmp = row1.get(filt) <= row2.get(filt)
            if not cmp:
                return False

        return True

    rows = sorted(rows, key=cmp_to_key(compare))
    return rows[0]


if __name__ == '__main__':
    main.cli()
