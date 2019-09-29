from asr.core import command, argument, option


def folderexists():
    from pathlib import Path
    assert Path('tree').is_dir()


tests = [
    {'cli': ['asr run setup.materials',
             'asr run "database.totree materials.json --run"'],
     'test': folderexists}
]


@command('asr.database.totree',
         tests=tests)
@argument('database', nargs=1)
@option('--run/--dry-run')
@option('-s', '--selection', help='ASE-DB selection')
@option('-t', '--tree-structure')
@option('--sort', help='Sort the generated materials '
        '(only useful when dividing chunking tree)')
@option('--data', is_flag=True, help='Unpack data')
@option('--copy', is_flag=True, help='Copy pointer tagged files')
@option('--atomsname', help='Filename to unpack atomic structure to')
@option('-c', '--chunks', metavar='N', help='Divide the tree into N chunks')
def main(database, run=False, selection='',
         tree_structure=('tree/{stoi}/{spg}/{formula:metal}-{stoi}-'
                         '{spg}-{wyck}-{uid}'),
         sort=None, data=False, atomsname='unrelaxed.json',
         chunks=1, copy=False):
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

    \b
    Examples:
    ---------
    For all these examples, suppose you have a database named "database.db".

    \b
    Unpack database using default parameters:
      asr run database.totree database.db --run
    \b
    Don't actually unpack the database but do a dry-run:
      asr run database.totree database.db
    \b
    Only select a part of the database to unpack:
      asr run database.totree database.db --selection "natoms<3" --run
    \b
    Set custom folder tree-structure:
      asr run database.totree database.db --tree-structure
          tree/{stoi}/{spg}/{formula:metal} --run
    \b
    Divide the tree into 2 chunks (in case the study of the materials)
    is divided between 2 people). Also sort after number of atoms,
    so computationally expensive materials are divided evenly:
      asr run database.totree database.db --sort natoms --chunks 2 --run
    """
    from os import makedirs
    from pathlib import Path
    from ase.db import connect
    from ase.io import write
    import spglib
    from asr.core import chdir, write_json, md5sum
    import importlib

    if selection:
        print(f'Selecting {selection}')

    if sort:
        print(f'Sorting after {sort}')

    assert Path(database).exists(), f'file: {database} doesn\'t exist'

    db = connect(database)
    rows = list(db.select(selection, sort=sort))

    folders = {}
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
        folders[row.id] = (folder, row)

    print(f'Number of collisions: {nc}')
    for er in err:
        print(er)

    if not run:
        print(f'Would make {len(folders)} folders')
        if chunks > 1:
            print(f'Would divide these folders into {chunks} chunks')

        print('The first 10 folders would be')
        for rowid, folder in folders.items()[:10]:
            print(f'    {folder}')
        print('    ...')
        print('To run the command use the --run option')
        return

    cwd = Path('.').absolute()
    for i, (rowid, (folder, row)) in enumerate(folders.items()):
        if chunks > 1:
            chunkno = i % chunks
            parts = list(Path(folder).parts)
            parts[0] += str(chunkno)
            folder = str(Path().joinpath(*parts))

        makedirs(folder)
        folder = Path(folder)
        with chdir(folder):
            write(atomsname, row.toatoms())
            if data:
                for filename, results in row.data.items():
                    # We treat json differently
                    if filename.endswith('.json'):
                        write_json(filename, results)
                    elif filename == '__links__':
                        for destdir, identifier in results.items():
                            destdir = Path(destdir).absolute()
                            try:
                                linkedrow = db.get(selection +
                                                   f',uid={identifier}')
                            except KeyError:
                                print(f'{folder}: Unknown unique identifier '
                                      f'{identifier}! Cannot link to'
                                      f' {destdir}.')
                                srcdir = None
                            else:
                                srcdir = cwd / folders[linkedrow.id][0]
                            finally:
                                destdir.symlink_to(srcdir,
                                                   target_is_directory=True)
                    elif results.get('pointer'):
                        path = results.get('pointer')
                        md5 = results.get('__md5__')
                        srcfile = Path(path)
                        if not srcfile.is_file():
                            print(f'Cannot locate source file: {path}')
                            continue
                        destfile = Path(filename)
                        if copy:
                            destfile.write_bytes(srcfile.read_bytes())
                        else:
                            destfile.symlink_to(srcfile)
                        if not md5sum(filename) == md5:
                            print('Warning: File {filename} does not match'
                                  ' original file. (Difference in '
                                  'checksums).')
                    else:
                        assert 'contents' in results, \
                            f'Unknown data entry {results}.'
                        contents = results.get('contents')
                        md5 = results.get('__md5__')
                        mod, func = results.get('write').split('@')

                        write = getattr(importlib.import_module(mod), func)
                        write(filename, contents)
                        if not md5sum(filename) == md5:
                            print('Warning: File {filename} does not match'
                                  ' original file. (Difference in '
                                  'checksums)')


if __name__ == '__main__':
    main.cli()
