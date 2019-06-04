from asr.utils import command, argument, option


@command('asr.setup.unpackdatabase',
         save_results_file=False)
@argument('database', nargs=1)
@option('--run/--dry-run', default=False)
@option('-s', '--selection', help='ASE-DB selection')
@option('-t', '--tree-structure',
        default=('tree/{stoi}/{spg}/{formula:metal}-{stoi}'
                 '-{spg}-{wyck}-{uid}'))
@option('--sort', help='Sort the generated materials '
        '(only useful when dividing chunking tree)')
@option('--kvp', is_flag=True, help='Unpack key-value-pairs')
@option('--data', is_flag=True, help='Unpack data')
@option('--atomsname', default='unrelaxed.json',
        help='Filename to unpack atomic structure to')
@option('-c', '--chunks', default=1, metavar='N',
        help='Divide the tree into N chunks')
def main(database, run, selection, tree_structure,
         sort, kvp, data, atomsname, chunks):
    """Set up folders with atomic structures based on ase-database"""
    from os import makedirs
    from pathlib import Path
    from ase.db import connect
    from ase.io import write
    import spglib
    from asr.utils import chdir, write_json

    # from ase import Atoms
    if selection:
        print(f'Selecting {selection}')

    if sort:
        print(f'Sorting after {sort}')

    db = connect(database)
    rows = list(db.select(selection, sort=sort))

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
        if chunks > 1:
            print(f'Would divide these folders into {chunks} chunks')

        print('The first 10 folders would be')
        for folder in folders[:10]:
            print(f'    {folder}')
        print('    ...')
        print('To run the command use the --run option')
        return

    for i, (folder, row) in enumerate(zip(folders, rows)):
        if chunks > 1:
            chunkno = i % chunks
            parts = list(Path(folder).parts)
            parts[0] += str(chunkno)
            folder = str(Path().joinpath(*parts))

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
