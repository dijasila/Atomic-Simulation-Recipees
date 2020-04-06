from asr.core import command, argument, option


@command('asr.database.filter')
@argument('databaseout', nargs=1)
@argument('databasein', nargs=1)
@option('-s', '--selection', help='Selection query.')
@option('-p', '--patterns',
        help='Comma separated string of patterns for filtering.')
def main(databasein, databaseout, selection='',
         patterns='*'):
    """Filter rows and data in database."""
    from ase.db import connect
    from pathlib import Path
    from fnmatch import fnmatch

    print(f'Filtering {databasein} into {databaseout}')
    patterns = patterns.split(',')
    dest = Path(databaseout)
    assert not dest.is_file(), \
        f'The destination path {databaseout} already exists.'

    def filter_func(key):
        return any(map(lambda pattern: fnmatch(key, pattern), patterns))

    with connect(databasein) as con, connect(databaseout) as conout:
        rows = con.select(selection)

        for row in rows:
            if patterns == '*':
                data = row.data
            else:
                keys = filter(filter_func, row.data)
                data = {key: row.data[key] for key in keys}

            conout.write(atoms=row.toatoms(), data=data, **row.key_value_pairs)
    # metadata = con.metadata
    conout.metadata = con.metadata


if __name__ == '__main__':
    main.cli()
