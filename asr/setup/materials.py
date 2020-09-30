"""Generate database with test systems."""
from asr.core import command, option


@command('asr.setup.materials',
         creates=['materials.json'])
@option('-s', '--selection', type=str,
        help='ASE DB selection string')
def main(selection: str = ''):
    """Create database with materials from the ASR materials database.

    The ASR materials database currently contains all elementary and
    binary materials from the Nomad benchmarking database.

    The created materials will be saved into the database
    "materials.json".

    Examples
    --------
    Get all materials from database
    $ asr run setup.materials
    In folder: . (1/1)
    Running asr.setup.materials(selection='')
    ...
    Created materials.json database containing 136 materials

    """
    from ase.db import connect
    from pathlib import Path

    dbname = str(Path(__file__).parent / 'testsystems.json')
    db = connect(dbname)
    rows = list(db.select(selection))

    nmat = len(rows)
    assert not Path('materials.json').is_file(), \
        'Database materials.json already exists!'

    print('PWD', Path().resolve())
    newdb = connect('materials.json')
    for row in rows:
        atoms = row.toatoms()
        kvp = row.key_value_pairs
        data = row.data
        newdb.write(atoms, key_value_pairs=kvp, data=data)
    print(f'Created materials.json database containing {nmat} materials')


group = 'setup'


if __name__ == '__main__':
    main.cli()
