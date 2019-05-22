from asr.utils import argument, option
import click


@click.command()
@argument('database', nargs=1)
@option('--run/--dry-run', default=False)
@option('-s', '--selection', help='ASE-DB selection')
@option('-t', '--tree-structure',
        default=('tree/{stoi}/{spg}/{formula:metal}-{stoi}-{spg}-{wyck}-{uid}'
                 '/{mag}'))
@option('--kvp', is_flag=True, help='Unpack key-value-pairs')
@option('--data', is_flag=True, help='Unpack data')
@option('--atomsname', default='unrelaxed.json',
        help='Filename to unpack atomic structure to')
def main(database, run, selection, tree_structure,
         kvp, data, atomsname):
    """Set up folders with atomic structures based on ase-database"""
    from os import makedirs
    from pathlib import Path
    from ase.db import connect
    from ase.io import write
    import spglib
    from asr.utils import chdir, write_json

    # from ase import Atoms
    if not selection:
        selection = ''
    db = connect(database)
    rows = list(db.select(selection))

    folders = []
    err = []
    nc = 0
    for row in rows:
        atoms = row.toatoms()
        formula = atoms.symbols.formula
        st = atoms.symbols.formula.stoichiometry()[0]
        cell = (atoms.cell.array,
                atoms.get_scaled_positions(),
                atoms.numbers)
        stoi = atoms.symbols.formula.stoichiometry()
        st = stoi[0]
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        sg = dataset['number']
        w = '-'.join(sorted(set(dataset['wyckoffs'])))
        if 'magstate' in row:
            magstate = row.magstate.lower()
        else:
            magstate = None

        # Add a unique identifier
        if 'uid' in tree_structure:
            for uid in range(0, 10):
                folder = tree_structure.format(stoi=st, spg=sg, wyck=w,
                                               formula=formula,
                                               mag=magstate,
                                               uid=uid)
                if folder not in folders:
                    break
            else:
                msg = ('Too many materials with same stoichiometry, '
                       'same space group and same formula')
                raise RuntimeError(msg)
            if uid > 0:
                nc += 1
                err += [f'Collision: {folder}']
        else:
            folder = tree_structure.format(stoi=st, spg=sg, wyck=w,
                                           formula=formula,
                                           mag=magstate)
        assert folder not in folders, f'{folder} already exists!'
        folders.append(folder)

    print(f'Number of collisions: {nc}')
    for er in err:
        print(er)

    if not run:
        print(f'Would make {len(folders)} folders')
        return
    
    for folder, row in zip(folders, rows):
        makedirs(folder)
        folder = Path(folder)
        with chdir(folder):
            write(atomsname, row.toatoms())
            if kvp:
                write_json('key-value-pairs.json', row.key_value_pairs)
            if kvp:
                for key in row.data:
                    write_json(f'{key}.json', row.data[key])


if __name__ == '__main__':
    main()
