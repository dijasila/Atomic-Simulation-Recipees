from asr.core import command, argument, option
from pathlib import Path


def make_folder_tree(*, folders, chunks,
                     write_atoms_file,
                     copy,
                     patterns,
                     atomsname,
                     create_folders):
    """Write folder tree to disk."""
    from os import makedirs, link
    from ase.io import write
    from asr.core import write_json
    import importlib
    from fnmatch import fnmatch

    cwd = Path('.').absolute()
    for i, (rowid, (folder, row)) in enumerate(folders.items()):
        if chunks > 1:
            chunkno = i % chunks
            parts = list(Path(folder).parts)
            parts[0] += str(chunkno)
            folder = str(Path().joinpath(*parts))

        folder = Path(folder)
        folder_has_been_created = False

        if write_atoms_file:
            if not folder_has_been_created:
                makedirs(folder)
                folder_has_been_created = True
            write(folder / atomsname, row.toatoms())

        for filename, results in row.data.items():
            for pattern in patterns:
                if fnmatch(filename, pattern):
                    break
            else:
                continue

            if not folder_has_been_created and create_folders:
                makedirs(folder)
                folder_has_been_created = True

            if (folder / filename).is_file():
                continue

            # We treat json differently
            if filename.endswith('.json'):
                write_json(folder / filename, results)

                # Unpack any extra files
                files = results.get('__files__', {})
                for extrafile, content in files.items():

                    if '__tofile__' in content:
                        tofile = content.pop('__tofile__')
                        mod, func = tofile.split('@')
                        write_func = getattr(importlib.import_module(mod),
                                             func)
                        write_func(folder / extrafile, content)
            elif filename == '__links__':
                pass
                # for destdir, identifier in results.items():
                #     if identifier not in folders:
                #         print(f'{folder}: Unknown unique identifier '
                #               f'{identifier}! Cannot link to'
                #               f' {destdir}.')
                #     else:
                #         srcdir = cwd / folders[identifier][0]
                #         (folder / destdir).symlink_to(srcdir,
                #                                       target_is_directory=True)
            else:
                path = results.get('pointer')
                srcfile = Path(path).resolve()
                if not srcfile.is_file():
                    print(f'Cannot locate source file: {path}')
                    continue
                destfile = folder / Path(filename)
                if destfile.is_file():
                    continue
                if copy:
                    try:
                        link(str(srcfile), str(destfile))
                    except OSError:
                        destfile.write_bytes(srcfile.read_bytes())
                else:
                    destfile.symlink_to(srcfile)


def make_folder_dict(rows, tree_structure):
    """Return a dictionary where key=uid and value=(folder, row)."""
    import spglib
    folders = {}
    folderlist = []
    err = []
    nc = 0
    child_uids = {}
    for row in rows:
        identifier = row.get('uid', row.id)
        links = row.data.get('__links__')
        if links:
            for path, link in links.items():
                assert link not in child_uids
                child_uids[link] = {'path': path, 'parentuid': identifier}

    for row in rows:
        identifier = row.get('uid', row.id)
        if identifier in child_uids:
            continue
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
        folder = tree_structure.format(stoi=st, spg=sg, wyck=w,
                                       formula=formula,
                                       mag=magstate,
                                       row=row)
        assert folder not in folderlist, f'Collision in folder: {folder}!'
        folderlist.append(folder)
        folders[identifier] = (folder, row)

    for child_uid, links in child_uids.items():
        parentfolder = folders[links['parentuid']]
        childfolder = str(Path().joinpaths(parentfolder, links['path']))
        folders[child_uid] = childfolder

    print(f'Number of collisions: {nc}')
    for er in err:
        print(er)
    return folders


@command('asr.database.totree')
@argument('database', nargs=1, type=str)
@option('--run/--dry-run', is_flag=True)
@option('-s', '--selection', help='ASE-DB selection', type=str)
@option('-t', '--tree-structure', type=str)
@option('--sort', help='Sort the generated materials '
        '(only useful when dividing chunking tree)', type=str)
@option('--copy/--no-copy', is_flag=True, help='Copy pointer tagged files')
@option('--atomsname', help='Filename to unpack atomic structure to', type=str)
@option('-c', '--chunks', metavar='N', help='Divide the tree into N chunks',
        type=int)
@option('--patterns',
        help="Comma separated patterns. Only unpack files matching patterns",
        type=str)
@option('--create-folders/--dont-create-folders', is_flag=True,
        help='Dont make new folders. Useful when writing to an existing tree.')
@option('--write-atoms-file/--dont-write-atoms-file', is_flag=True,
        help='Write atoms object to file with name given '
        'by the --atomsname option')
def main(database: str, run: bool = False, selection: str = '',
         tree_structure: str = ('tree/{stoi}/{spg}/{formula:metal}-{stoi}-'
                                '{spg}-{wyck}-{uid}'),
         sort: str = None, atomsname: str = 'structure.json',
         chunks: int = 1, copy: bool = False,
         patterns: str = '*', create_folders: bool = True,
         write_atoms_file: bool = True):
    """Unpack an ASE database to a tree of folders.

    This setup recipe can unpack an ASE database to into folders
    that have a tree like structure where directory names can be
    given by the material parameters such stoichiometry, spacegroup
    number for example: stoichiometry/spacegroup/formula.

    The specific tree structure is given by the --tree-structure
    option which can be customized according to the following table

    * {stoi}: Material stoichiometry
    * {spg}: Material spacegroup number
    * {formula}: Chemical formula. A possible variant is {formula:metal}
      in which case the formula will be sorted by metal atoms
    * {wyck}: Unique wyckoff positions. The unique alphabetically
      sorted Wyckoff positions.
    * {uid}: This is a unique identifier which starts at 0 and adds 1 if
      collisions (cases where two materials would go to the same folder)
      occur. In practice, if two materials would be unpacked in A-0/
      they would now be unpacked in A-0/ and A-1/.

    By default, the atomic structures will be saved into an unrelaxed.json
    file which is be ready to be relaxed. This filename can be changed with
    the --atomsname switch.

    Examples
    --------
    For all these examples, suppose you have a database named "database.db".

    Unpack database using default parameters:
    >>> asr run "database.totree database.db --run"
    Don't actually unpack the database but do a dry-run:
    >>> asr run "database.totree database.db"
    Only select a part of the database to unpack:
    >>> asr run "database.totree database.db --selection natoms<3 --run"
    Set custom folder tree-structure:
    >>> asr run "database.totree database.db
    >>> ... --tree-structure tree/{stoi}/{spg}/{formula:metal} --run"

    Divide the tree into 2 chunks (in case the study of the materials)
    is divided between 2 people). Also sort after number of atoms,
    so computationally expensive materials are divided evenly:
    >>> asr run "database.totree database.db --sort natoms --chunks 2 --run"
    """
    from pathlib import Path
    from ase.db import connect

    if selection:
        print(f'Selecting {selection}')

    if sort:
        print(f'Sorting after {sort}')

    assert Path(database).exists(), f'file: {database} doesn\'t exist'

    db = connect(database)
    rows = list(db.select(selection, sort=sort))

    patterns = patterns.split(',')
    folders = make_folder_dict(rows, tree_structure)

    if not run:
        print(f'Would (at most) make {len(folders)} folders')
        if chunks > 1:
            print(f'Would divide these folders into {chunks} chunks')

        print('The first 10 folders would be')
        for rowid, folder in list(folders.items())[:10]:
            print(f'    {folder[0]}')
        print('    ...')
        print('To run the command use the --run option')
        return

    make_folder_tree(folders=folders, chunks=chunks,
                     write_atoms_file=write_atoms_file, copy=copy,
                     patterns=patterns, atomsname=atomsname,
                     create_folders=create_folders)


if __name__ == '__main__':
    main.cli()
