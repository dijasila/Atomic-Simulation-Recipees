from asr.core import command, option


@command(creates=['materials.json'])
@option('-s', '--selection', type=str,
        help='ASE DB selection string')
def main(selection=''):
    """Create database with materials from the ASR materials database.

    The ASR materials database currently contains all elementary and
    binary materials from the Nomad benchmarking database.

    The created materials will be saved into the database
    "materials.json".

    \b
    Examples:
    ---------
    Get all materials from database with ntypes=1 (elementary compounds)
        asr run "setup.materials -s ntypes=1"
    """
    from ase.db import connect
    from pathlib import Path

    dbname = str(Path(__file__).parent / 'testsystems.json')
    db = connect(dbname)
    rows = list(db.select(selection))

    nmat = len(rows)
    assert not Path('materials.json').is_file(), \
        'Database materials.json already exists!'

    with connect('materials.json') as newdb:
        for row in rows:
            atoms = row.toatoms()
            kvp = row.key_value_pairs
            data = row.data
            newdb.write(atoms, key_value_pairs=kvp, data=data)
    print(f'Created materials.json database containing {nmat} materials')


if __name__ == '__main__':
    main.cli()
