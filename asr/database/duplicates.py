from asr.core import command, argument, option, ASRResult, ASEDatabase
from datetime import datetime


@command(module='asr.database.duplicates')
@argument('databaseout', type=ASEDatabase(), required=False)
@argument('database', type=ASEDatabase())
@option('-f', '--filterstring',
        help='List of keys denoting the priority of picking'
        ' candidates among possible duplicates.',
        type=str)
@option('-c', '--comparison-keys',
        help='Keys that have to be identical for materials to be identical.',
        type=str)
@option('-r', '--rmsd-tol', help='RMSD tolerance.', type=float)
@option('--skip-distance-calc', default=False, is_flag=True,
        help="Skip distance calculation. Only match structures "
        "based on their reduced formula and comparison_keys.")
def main(database: str,
         databaseout: str = None,
         filterstring: str = '<=natoms,<energy',
         comparison_keys: str = '',
         rmsd_tol: float = 0.3,
         skip_distance_calc: bool = False) -> ASRResult:
    """Filter out duplicates of a database.

    Parameters
    ----------
    database : db-connection
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
    skip_distance_calc : bool
        If true, only use reduced formula and comparison_keys to match
        structures. Skip calculating distances between structures. The
        output rmsd's will be 0 for matching structures.

    Returns
    -------
    dict
        Keys:
            - ``duplicate_groups``: Dict containing all duplicate groups.
              The key of each group is the uid of the prioritized candidate
              of the group.

    """
    # from ase.db import connect
    from asr.database.rmsd import main as rmsd
    from asr.utils import timed_print
    assert database != databaseout, \
        'You cannot read and write from the same database.'

    ops_and_keys = parse_filter_string(filterstring)

    # if not rmsd.done:
    rmsd_results = rmsd(database, comparison_keys=comparison_keys,
                        skip_distance_calc=skip_distance_calc)
    # rmsd_results = read_json('results-asr.database.rmsd.json')
    rmsd_by_id = rmsd_results['rmsd_by_id']
    uid_key = rmsd_results['uid_key']
    duplicate_groups = []
    # db = connect(database)
    exclude_uids = set()
    already_checked_uids = set()
    nrmsd = len(rmsd_by_id)
    rows = {}

    for row in database.select(include_data=False):
        rows[row.get(uid_key)] = row

    print('Filtering materials...')
    for irmsd, (uid, rmsd_dict) in enumerate(rmsd_by_id.items()):
        if uid in already_checked_uids:
            continue
        now = datetime.now()
        timed_print(f'{now:%H:%M:%S}: {irmsd}/{nrmsd}', wait=30)
        duplicate_uids = find_duplicate_group(uid, rmsd_by_id, rmsd_tol)

        # Pick the preferred row according to filterstring
        include = filter_uids(rows, duplicate_uids,
                              ops_and_keys, uid_key)
        # Book keeping
        already_checked_uids.update(duplicate_uids)

        exclude = duplicate_uids - include
        if exclude:
            exclude_uids.update(exclude)
            duplicate_groups.append({'exclude': list(exclude),
                                     'include': list(include)})

    if databaseout is not None:
        nmat = len(rows)
        # with connect(databaseout) as filtereddb:
        for uid, row in rows.items():
            now = datetime.now()
            timed_print(f'{now:%H:%M:%S}: {row.id}/{nmat}', wait=30)

            if uid in exclude_uids:
                continue
            databaseout.write(atoms=row.toatoms(),
                              data=row.data,
                              **row.key_value_pairs)

        databaseout.metadata = database.metadata

    filterkeys = [key for _, key in ops_and_keys]
    for ig, group in enumerate(duplicate_groups):
        include = group['include']
        exclude = group['exclude']
        max_rmsd = 0
        for uid in include + exclude:
            max_rmsd = max([max_rmsd,
                            max(value for value in rmsd_by_id[uid].values()
                                if value is not None and value < rmsd_tol)])
        print(f'Group #{ig} max_rmsd={max_rmsd}')
        print('    Excluding:')
        for uid in exclude:
            row = rows[uid]
            print(f'        {uid} '
                  + ' '.join(f'{key}=' + str(row.get(key)) for key in filterkeys))
        print('    Including:')
        for uid in include:
            row = rows[uid]
            print(f'        {uid} '
                  + ' '.join(f'{key}=' + str(row.get(key)) for key in filterkeys))

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


def filter_uids(all_rows, duplicate_ids, ops_and_keys, uid_key):
    """Get most important rows according to filterstring.

    Parameters
    ----------
    all_rows: dict
        Dictionary with key=uid and value=row.
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
    rows = [all_rows[uid] for uid in duplicate_ids]

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


def find_duplicate_group(uid, rmsd_by_id, rmsd_tol,
                         already_considered_uids=None):
    if already_considered_uids is None:
        already_considered_uids = {uid}
    else:
        already_considered_uids.add(uid)

    duplicate_uids = set(key for key, value in rmsd_by_id[uid].items()
                         if value is not None and value < rmsd_tol)
    new_uids = duplicate_uids - already_considered_uids
    if new_uids:
        for new_uid in new_uids:
            find_duplicate_group(new_uid, rmsd_by_id, rmsd_tol,
                                 already_considered_uids=already_considered_uids)

    return already_considered_uids


if __name__ == '__main__':
    main.cli()
