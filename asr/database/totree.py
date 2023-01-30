"""Convert database to folder tree."""
from asr.core import command, argument, option, ASRResult
from asr.utils import timed_print
from pathlib import Path
from datetime import datetime


def make_folder_tree(*, folders, chunks,
                     copy,
                     patterns,
                     atomsfile,
                     update_tree):
    """Write folder tree to disk."""
    from os import makedirs, link
    from ase.io import write
    from asr.core import write_json
    import importlib
    from fnmatch import fnmatch

    nfolders = len(folders)
    for i, (rowid, (folder, row)) in enumerate(folders.items()):
        now = datetime.now()
        percentage_completed = (i + 1) / nfolders * 100
        timed_print(f'{now:%H:%M:%S} {i + 1}/{nfolders} '
                    f'{percentage_completed:.1f}%', wait=30)
        if chunks > 1:
            chunkno = i % chunks
            parts = list(Path(folder).parts)
            parts[0] += str(chunkno)
            folder = str(Path().joinpath(*parts))

        folder = Path(folder)

        if not update_tree and atomsfile:
            if not folder.is_dir():
                makedirs(folder)
            write(folder / atomsfile, row.toatoms())

        for filename, results in row.data.items():
            for pattern in patterns:
                if fnmatch(filename, pattern):
                    break
            else:
                continue

            if not folder.is_dir():
                if not update_tree:
                    makedirs(folder)
                else:
                    continue

            if (folder / filename).is_file() and not update_tree:
                continue

            # We treat json differently
            if filename.endswith('.json'):
                write_json(folder / filename, results)
            elif filename == '__children__':
                # Some silly mechanism to have subfolders like in stiffness
                pass
            else:
                raise RuntimeError('Non-JSON file handling removed after '
                                   '34f9c5016d15e9a965f867eada7f756f4f0fbc34:'
                                   f'{filename}')


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
        children = row.data.get('__children__')
        if children:
            for path, child_uid in children.items():
                if child_uid in child_uids:
                    existing_path = child_uids[child_uid]['path']
                    assert (existing_path.startswith(path)
                            or path.startswith(existing_path))
                    if path.startswith(existing_path):
                        continue
                child_uids[child_uid] = {'path': path, 'parentuid': identifier}

    for row in rows:
        identifier = row.get('uid', row.id)
        if identifier in child_uids:
            folders[identifier] = (None, row)
            continue
        atoms = row.toatoms()
        formula = atoms.symbols.formula
        st = atoms.symbols.formula.stoichiometry()[0]
        cell = (atoms.cell.array,
                atoms.get_scaled_positions(),
                atoms.numbers)
        stoi = atoms.symbols.formula.stoichiometry()
        st = stoi[0]
        reduced_formula = stoi[1]
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
                                       reduced_formula=reduced_formula,
                                       formula=formula,
                                       mag=magstate,
                                       row=row)
        assert folder not in folderlist, f'Collision in folder: {folder}!'
        folderlist.append(folder)
        folders[identifier] = (folder, row)

    for child_uid, links in child_uids.items():
        parent_uid = links['parentuid']
        if child_uid not in folders:
            print(f'Parent (uid={parent_uid}) has unknown child '
                  f'(uid={child_uid}).')
            continue
        parentfolder = folders[parent_uid][0]
        childfolder = str(Path().joinpath(parentfolder, links['path']))
        folders[child_uid] = (childfolder, folders[child_uid][1])

    print(f'Number of collisions: {nc}')
    for er in err:
        print(er)
    return folders


@command('asr.database.totree',
         save_results_file=False)
@argument('database', nargs=1, type=str)
@option('--run/--dry-run', is_flag=True)
@option('-s', '--selection', help='ASE-DB selection', type=str)
@option('-t', '--tree-structure', type=str)
@option('--sort', help='Sort the generated materials '
        '(only useful when dividing chunking tree)', type=str)
@option('--copy/--no-copy', is_flag=True, help='Copy pointer tagged files')
@option('--atomsfile',
        help="Filename to unpack atomic structure to. "
        "By default, don't write atoms file.",
        type=str)
@option('-c', '--chunks', metavar='N', help='Divide the tree into N chunks',
        type=int)
@option('--patterns',
        help="Comma separated patterns. Only unpack files matching patterns",
        type=str)
@option('--update-tree', is_flag=True,
        help='Update results files in existing folder tree.')
def main(database: str, run: bool = False, selection: str = '',
         tree_structure: str = (
             'tree/{stoi}/{reduced_formula:abc}/{row.uid}'
         ),
         sort: str = None, atomsfile: str = None,
         chunks: int = 1, copy: bool = False,
         patterns: str = '*', update_tree: bool = False) -> ASRResult:
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
    * {reduced_formula}: Reduced chemical formula. Like {formula}
      except the formula has been reduced, i.e., Mo2S4 -> MoS2.
    * {wyck}: Unique wyckoff positions. The unique alphabetically
      sorted Wyckoff positions.

    Examples
    --------
    For all these examples, suppose you have a database named "database.db".

    Unpack database using default parameters:
    $ asr run "database.totree database.db --run"
    Don't actually unpack the database but do a dry-run:
    $ asr run "database.totree database.db"
    Only select a part of the database to unpack:
    $ asr run "database.totree database.db --selection natoms<3 --run"
    Set custom folder tree-structure:
    $ asr run "database.totree database.db
    ... --tree-structure tree/{stoi}/{spg}/{formula:metal} --run"

    Divide the tree into 2 chunks (in case the study of the materials)
    is divided between 2 people). Also sort after number of atoms,
    so computationally expensive materials are divided evenly:
    $ asr run "database.totree database.db --sort natoms --chunks 2 --run"

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
                     atomsfile=atomsfile, copy=copy,
                     patterns=patterns,
                     update_tree=update_tree)


if __name__ == '__main__':
    main.cli()
