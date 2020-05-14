from asr.core import command, argument, option
from datetime import datetime


@command(module='asr.database.duplicates',
         resources='1:20m',
         save_results_file=False)
@argument('databaseout')
@argument('database')
@option('-f', '--filterstring',
        help='List of keys denoting the priority of picking'
        ' a candidate among duplicates. Preface with + if '
        'you want to prioritize larger values.')
@option('-c', '--comparison-keys',
        help='Keys that have to be identical for materials to be identical.')
@option('-r', '--rmsd-tol', help='RMSD tolerance.')
def main(database, databaseout,
         filterstring='natoms,id', comparison_keys='',
         rmsd_tol=0.3):
    """Filter out duplicates of a database.

    Parameters
    ----------
    database : str
        Database to be analyzed for duplicates.
    databaseout : str
        Filename of new database with duplicates removed.
    filterstring : str
        Comma separated string of to keys determining priority of
        picking of row. Preface key with '+' to prioritize larger
        values.
    comparison_keys : str
        Comma separated string of keys that should be identical
        between rows to be compared. Eg. 'magstate,natoms'. Default is
        'natoms,id' which would first prioritize picking the structure
        with fewest atoms and then picking the one with the smallest
        id.
    rmsd_tol : float
        Tolerance on RMSD between materials for them to be considered
        to be duplicates.

    Returns
    -------
    dict
        Keys:
            - ``duplicate_groups``: Dict containing all duplicate groups.
              The key of each group is the uid of the prioritized candidate
              of the group.

    """
    from ase.db import connect
    from asr.core import read_json
    from asr.database.rmsd import main as rmsd
    from asr.database.rmsd import _timed_print
    assert database != databaseout, \
        'You cannot read and write from the same database.'

    if not rmsd.done:
        rmsd(database, comparison_keys=comparison_keys)
    rmsd_results = read_json('results-asr.database.rmsd.json')
    rmsd_by_id = rmsd_results['rmsd_by_id']
    uid_key = rmsd_results['uid_key']
    duplicate_groups = {}
    db = connect(database)
    exclude_uids = set()
    already_checked_uids = set()

    for uid, rmsd_dict in rmsd_by_id.items():
        if uid in already_checked_uids:
            continue
        duplicate_uids = set(key for key, value in rmsd_dict.items()
                             if value is not None and value < rmsd_tol)
        duplicate_uids.add(uid)

        # Pick the preferred row according to filterstring
        preferred_row = pick_out_row(db, duplicate_uids, filterstring, uid_key)
        preferred_uid = preferred_row.get(uid_key)

        # Book keeping
        already_checked_uids.update(duplicate_uids)

        exclude = duplicate_uids - {preferred_uid}
        if exclude:
            exclude_uids.update(exclude)
            duplicate_groups[preferred_uid] = list(duplicate_uids)

    comparison_keys = comparison_keys.split(',')
    nmat = len(db)
    with connect(databaseout) as filtereddb:
        for row in db.select():
            now = datetime.now()
            _timed_print(f'{now:%H:%M:%S}: {row.id}/{nmat}', wait=30)

            if row.get(uid_key) in exclude_uids:
                continue
            filtereddb.write(atoms=row.toatoms(),
                             data=row.data,
                             **row.key_value_pairs)

    filtereddb.metadata = db.metadata

    for preferred_uid, group in duplicate_groups.items():
        print(f'Chose {uid_key}={preferred_uid} out of')
        print('    ', ', '.join([str(item) for item in group]))

    print(f'Excluded {len(exclude_uids)} materials.')
    return {'duplicate_groups': duplicate_groups}


def pick_out_row(db, duplicate_ids, filterstring, uid_key):
    rows = [db.get(f'{uid_key}={uid}') for uid in duplicate_ids]
    keys = filterstring.split(',')

    reverses = []
    for i, key in enumerate(keys):
        if key.startswith('+'):
            reverse = True
            keys[i] = key[1:]
        else:
            reverse = False
        reverses.append(reverse)

    def keyfunc(row):
        values = []
        for key, reverse in zip(keys, reverses):
            value = row.get(key)
            if value is None:
                values.append(value)
                continue

            if reverse:
                values.append(-value)
            else:
                values.append(value)
        return tuple(values)

    rows = sorted(rows, key=keyfunc)
    return rows[0]


if __name__ == '__main__':
    main.cli()
