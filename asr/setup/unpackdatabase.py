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
@option('--copy', is_flag=True, default=False,
        help='Copy pointer tagged files')
def main(database, run, selection, tree_structure,
         sort, kvp, data, atomsname, chunks, copy):
    """Unpack an ASE database to a tree of folders.

    This setup recipe can unpack an ASE database to into folders
    that have a tree like structure where directory names can be
    given by the material parameters such stoichiometry, spacegroup
    number for example: stoichiometry/spacegroup/formula.

    The specific tree structure is given by the --tree-structure
    option which can be customized according to the following table

    \b
    {stoi}: Material stoichiometry
    {spg}: Material spacegroup number
    {formula}: Chemical formula. A possible variant is {formula:metal}
        in which case the formula will be sorted by metal atoms
    {wyck}: Unique wyckoff positions. The unique alphabetically sorted
        Wyckoff positions.
    {uid}: This is a unique identifier which starts at 0 and adds 1 if
        collisions (cases where two materials would go to the same folder)
        occur. In practice, if two materials would be unpacked in A-0/
        they would now be unpacked in A-0/ and A-1/.

    By default, the atomic structures will be saved into an unrelaxed.json
    file which is be ready to be relaxed. This filename can be changed with
    the --atomsname switch.

    If the database contains some interesting data and key-value-pairs
    these can also be unpacked with the --kvp and --data switches,
    respectively. The key-value-pairs will be saved to key-value-pairs.json
    and each key in data will be saved to "key.json"

    \b
    Examples:
    ---------
    For all these examples, suppose you have a database named "database.db".

    \b
    Unpack database using default parameters:
      asr run setup.unpackdatabase database.db --run
    \b
    Don't actually unpack the database but do a dry-run:
      asr run setup.unpackdatabase database.db
    \b
    Only select a part of the database to unpack:
      asr run setup.unpackdatabase database.db --selection "natoms<3" --run
    \b
    Set custom folder tree-structure:
      asr run setup.unpackdatabase database.db --tree-structure
          tree/{stoi}/{spg}/{formula:metal} --run
    \b
    Divide the tree into 2 chunks (in case the study of the materials)
    is divided between 2 people). Also sort after number of atoms,
    so computationally expensive materials are divided evenly:
      asr run setup.unpackdatabase database.db --sort natoms --chunks 2 --run
    \b
    Unpack key-value-pairs and data keys of the ASE database as well:
      asr run setup.unpackdatabase database.db --kvp --data --run
    """
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
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3,
                                              angle_tolerance=0.1)
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
            if data:
                for key, results in row.data.items():
                    write_json(f'{key}.json', results)
                    if not isinstance(results, dict):
                        continue
                    kd = results.get('__key_descriptions__', {})
                    for rkey in results:
                        if rkey in kd and kd[rkey].startswith('File:'):
                            srcfile = Path(results[rkey])
                            destfile = Path(rkey)
                            if copy:
                                destfile.write_bytes(srcfile.read_bytes())
                            else:
                                destfile.symlink_to(srcfile)


if __name__ == '__main__':
    main()
