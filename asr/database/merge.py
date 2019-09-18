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
    dbmerged = connect(databaseout)

    # First merge rows common in both databases
    with connect(databaseout) as dbmerged:
        for row1 in db1.select():
            structid = row1.get(identifier)
            matching = list(db2.select(f'{identifier}={structid}'))

            if len(matching) > 1:
                raise RuntimeError(f'More than one structure in {database2} '
                                   f'matching identifier={identifier}')
            elif len(matching) == 0:
                print(f'No matching structures {row1.formula}')
                continue
            row2 = matching[0]

            kvp1 = row1.key_value_pairs
            kvp2 = row2.key_value_pairs
            kvpmerged = kvp2.copy().update(kvp1)

            data1 = row1.data
            data2 = row2.data
            datamerged = data2.copy().update(data1)
            dbmerged.write(row1.toatoms(), key_value_pairs=kvpmerged,
                           data=datamerged)
    
        # Then insert rows that only exist in one of the databases
        for row in db1.select():
            structid = row.get(identifier)
            matching = list(db2.select(f'{identifier}={structid}'))
            if not len(matching):
                dbmerged.write(row.toatoms(),
                               key_value_pairs=row.key_value_pairs,
                               data=row.data)

        for row in db2.select():
            structid = row.get(identifier)
            matching = list(db1.select(f'{identifier}={structid}'))
            if not len(matching):
                dbmerged.write(row.toatoms(),
                               key_value_pairs=row.key_value_pairs,
                               data=row.data)


if __name__ == '__main__':
    main.cli()
