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

    ops_and_keys = parse_filter_string(filterstring)

    if not rmsd.done:
        rmsd(database, comparison_keys=comparison_keys)
    rmsd_results = read_json('results-asr.database.rmsd.json')
    rmsd_by_id = rmsd_results['rmsd_by_id']
    uid_key = rmsd_results['uid_key']
    duplicate_groups = []
    db = connect(database)
    exclude_uids = set()
    already_checked_uids = set()
    nrmsd = len(rmsd_by_id)
    print('Filtering materials...')
    for irmsd, (uid, rmsd_dict) in enumerate(rmsd_by_id.items()):
        if uid in already_checked_uids:
            continue
        now = datetime.now()
        timed_print(f'{now:%H:%M:%S}: {irmsd}/{nrmsd}', wait=30)
        duplicate_uids = set(key for key, value in rmsd_dict.items()
                             if value is not None and value < rmsd_tol)
        duplicate_uids.add(uid)

        # Pick the preferred row according to filterstring
        include = filter_uids(db, duplicate_uids,
                              ops_and_keys, uid_key)
        # Book keeping
        already_checked_uids.update(duplicate_uids)

        exclude = duplicate_uids - include
        if exclude:
            exclude_uids.update(exclude)
            duplicate_groups.append({'exclude': list(exclude),
                                     'include': list(include)})

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

    filterkeys = [key for _, key in ops_and_keys]
    for ig, group in enumerate(duplicate_groups):
        include = group['include']
        exclude = group['exclude']
        print(f'Group #{ig}')
        print('    Excluding:')
        for uid in exclude:
            row = db.get(f'{uid_key}={uid}')
            print(f'        {uid} ' + ' '.join(f'{key}=' + str(row.get(key))
                                               for key in filterkeys))
        print('    Including:')
        for uid in include:
            row = db.get(f'{uid_key}={uid}')
            print(f'        {uid} ' + ' '.join(f'{key}=' + str(row.get(key))
                                               for key in filterkeys))

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


def filter_uids(db, duplicate_ids, ops_and_keys, uid_key):
    """Get most important rows according to filterstring.

    Parameters
    ----------
    db: Database connection
        Open database connection.
    duplicate_ids: iterable
        Set of possible duplicate materials.
    ops_and_keys: List[Tuple(str, str)]
        List of filters where the first element of the tuple is the comparison
        operator and the second is the to compare i.e.: [('<',
        'energy')]. Other accepted operators are {'<=', '>=', '>', '<', '=='}.
        Additional filters can be added to construct more complex filters,
        i.e., `[('<', 'energy'), ('<=', 'natoms')]` means that a material is
        only picked if no other materials with lower energy AND fewer or same
        number of atoms exists.
    uid_key: str
        The UID key of the database connection which the duplicate_ids
        parameters are refererring to.

    Returns
    -------
    filtered_uids: `set`
        Set of filtered uids.

    """
    rows = [db.get(f'{uid_key}={uid}') for uid in duplicate_ids]

    filtered_uids = set()
    for candidaterow in rows:
        better_candidates = {
            row for row in rows
            if all(compare(row[key], candidaterow[key], op)
                   for op, key in ops_and_keys)}
        if not better_candidates:
            filtered_uids.add(candidaterow.get(f'{uid_key}'))

    return filtered_uids


def parse_filter_string(filterstring):
    """Parse a comma separated filter string.

    Parameters
    ----------
    filterstring: str
        Comma separated filter string, i.e. '<energy,<=natoms'

    Returns
    -------
    ops_and_keys: List[Tuple(str, str)]
        For the above example would return [('<', 'energy'), ('<=', 'natoms')].

    """
    filters = filterstring.split(',')
    sorts = ['<=', '>=', '==', '>', '<']
    ops_and_keys = []
    for filt in filters:
        for op in sorts:
            if filt.startswith(op):
                break
        else:
            raise ValueError(f'Unknown sorting operator in filterstring={filt}.')
        key = filt[len(op):]
        ops_and_keys.append((op, key))
    return ops_and_keys


if __name__ == '__main__':
    main.cli()
