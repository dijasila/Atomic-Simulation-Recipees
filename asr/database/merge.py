from asr.core import command, argument, option


def folderexists():
    from pathlib import Path
    assert Path('tree').is_dir()


tests = [
    {'cli': ['asr run setup.materials',
             'asr run database.totree materials.json --run'],
     'test': folderexists}
]


@command('asr.database.totree',
         tests=tests)
@argument('databaseout', nargs=1)
@argument('database2', nargs=1)
@argument('database1', nargs=1)
@option('--identifier')
def main(database1, database2, databaseout, identifier='asr_id'):
    """Merge two ASE databases."""
    from ase.db import connect
    import numpy as np

    db1 = connect(database1)
    db2 = connect(database2)
    dbout = connect(databaseout)

    column1 = set(db1.columnnames)
    column2 = set(db2.columnnames)

    commoncolumns = column1.union(column2)

    for row1 in db1.select():
        print(row1.formula)
        structid = row1.get(identifier)
        row2 = db2.get(f'{identifier}={structid}')

        skip_keys = ['ctime', 'mtime', 'unique_id', 'uid', 'user']
        for key in row1:
            if key not in row2 and key not in skip_keys:
                continue

            val1 = row1[key]
            val2 = row2[key]
            assert type(val1) == type(val2), 'type({key1}) != type({key2})'
            if isinstance(val1, np.ndarray):
                valf1 = val1.flatten().astype(float)
                valf2 = val2.flatten().astype(float)
                assert (valf1 - valf2 == 0.0).all(), \
                    f'{row1[key]} != {row2[key]}'
            else:
                assert row1[key] == row2[key], f'{row1[key]} != {row2[key]}'

        # print(row1.toatoms())
        # print(row1.key_value_pairs)
        # sel = ','.join([f'{key}={row1.get(key)}' for key in commoncolumns
        #                 if key in row1.key_value_pairs])
        # # sel = f'id={row1.unique_id}'
        
        # print(sel)
        # exit()
        # row2 = db2.select(sel)


if __name__ == '__main__':
    main.cli()
