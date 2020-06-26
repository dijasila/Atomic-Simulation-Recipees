from asr.core import command, argument, option
from datetime import datetime


@command(module='asr.database.duplicates',
         resources='1:20m',
         save_results_file=False)
@argument('databaseout', type=str, required=False)
@argument('database', type=str)
@option('-f', '--filterstring',
        help='List of keys denoting the priority of picking'
        ' candidates among possible duplicates.',
        type=str)
@option('-c', '--comparison-keys',
        help='Keys that have to be identical for materials to be identical.',
        type=str)
@option('-r', '--rmsd-tol', help='RMSD tolerance.', type=float)
def main(database: str,
         databaseout: str = None,
         filterstring: str = '<=natoms,<energy',
         comparison_keys: str = '',
         rmsd_tol: float = 0.3):
    """Filter out duplicates of a database.

    Parameters
    ----------
    database : str
        Database to be analyzed for duplicates.
    databaseout : str
        Filename of new database with duplicates removed.
    filterstring : str
        Comma separated string of filters. A simple filter could be '<energy'
        which only pick a material if no other material with lower energy
        exists (in other words: chose the lowest energy materials). '<' means
        'smallest'. Other accepted operators are {'<=', '>=', '>', '<', '=='}.
        Additional filters can be added to construct more complex filters,
        i.e., '<energy,<=natoms' means that a material is only picked if no
        other materials with lower energy AND fewer or same number of atoms
        exists.
    comparison_keys : str
        Comma separated string of keys that should be identical
        between rows to be compared. Eg. 'magstate,natoms'.
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
    from asr.utils import timed_print
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
        preferred_rows = pick_rows_according_to_filter(db, duplicate_uids,
                                                       filterstring, uid_key)
        preferred_uids = {preferred_row.get(uid_key)
                          for preferred_row in preferred_rows}

        # Book keeping
        already_checked_uids.update(duplicate_uids)

        exclude = duplicate_uids - preferred_uids
        if exclude:
            exclude_uids.update(exclude)
            dup_uids = list(duplicate_uids)
            for preferred_uid in preferred_uids:
                duplicate_groups[preferred_uid] = dup_uids

    if databaseout is not None:
        nmat = len(db)
        with connect(databaseout) as filtereddb:
            for row in db.select():
                now = datetime.now()
                timed_print(f'{now:%H:%M:%S}: {row.id}/{nmat}', wait=30)

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
    return {'duplicate_groups': duplicate_groups,
            'duplicate_uids': list(exclude_uids)}


def compare(value1, value2, comparator):
    """Return value1 {comparator} value2."""
    if comparator == '<=':
        return value1 <= value2
    elif comparator == '>=':
        return value1 >= value2
    elif comparator == '<':
        return value1 < value2
    elif comparator == '>':
        return value1 > value2
    elif comparator == '==':
        return value1 == value2


def pick_rows_according_to_filter(db, duplicate_ids, filterstring, uid_key):
    """Get most important rows according to filterstring.

    Parameters
    ----------
    db: Database connection
        Open database connection.
    duplicate_ids: iterable
        Set of possible duplicate materials.
    filterstring: str
        Comma separated string of filters. A simple filter could be '<energy'
        which only pick a material if no other material with lower energy
        exists (in other words: chose the lowest energy materials). '<' means
        'smallest'. Other accepted operators are {'<=', '>=', '>', '<', '=='}.
        Additional filters can be added to construct more complex filters,
        i.e., '<energy,<=natoms' means that a material is only picked if no
        other materials with lower energy AND fewer or same number of atoms
        exists.
    uid_key: str
        The UID key of the database connection which the duplicate_ids
        parameters are refererring to.

    Returns
    -------
    picked_rows: `list`
        List of filtered rows.

    """
    rows = [db.get(f'{uid_key}={uid}') for uid in duplicate_ids]
    filters = filterstring.split(',')
    sorts = {'<=', '>=', '==', '>', '<'}
    ops_and_keys = []
    for filt in filters:
        for op in sorts:
            if filt.startswith(op):
                break
        else:
            raise ValueError(f'Unknown sorting operator in filterstring={filt}.')
        key = filt[len(op):]
        ops_and_keys.append((op, key))

    picked_rows = []
    for candidaterow in rows:
        better_candidates = {
            row for row in rows
            if all(compare(row[key], candidaterow[key], op)
                   for op, key in ops_and_keys)}
        if not better_candidates:
            picked_rows.append(candidaterow)

    return picked_rows


if __name__ == '__main__':
    main.cli()
