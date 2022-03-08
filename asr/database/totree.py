"""Convert database to folder tree."""
from asr.core import ASRResult
from asr.utils import timed_print
from pathlib import Path
from datetime import datetime


def make_folder_tree(*, repo, folders, chunks,
                     copy,
                     patterns,
                     update_tree):
    """Write folder tree to disk."""
    from os import makedirs
    from ase.io import write

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
        atoms = row.toatoms()
        yield atoms, folder

    return

    # Disable old loop code temporarily:
    for nothing in []:
        # XXX old code to unpack trees with records in them.
        # We need to rehabilitate this as well.
        records = row.records

        from asr.core import get_cache, chdir

        if not records:
            continue

        if not folder.is_dir():
            if not update_tree:
                makedirs(folder)
            else:
                continue

        with chdir(folder):
            cache = get_cache()
            for record in records:
                cache.add(record)


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


# XXX Has the effect of "tagging" the atoms as a task without actually
# calculating anything.  Should be changed so htw framework can do this
# "tagging" in a less roundabout way.
def structure(atoms):
    from ase.io import write
    # We write the atoms to a file in ASE format so it can be inspected
    # in ASE GUI.
    path = Path('atoms.json')
    write(path, atoms)
    return atoms


def main(
        database: str, run: bool = False, select: str = '',
        tree_structure: str = '{stoi}/{reduced_formula:abc}',
        # sort: str = None,
        # chunks: int = 1, copy: bool = False,
        #patterns: str = '*', update_tree: bool = False
) -> ASRResult:
    from htwutil.repository import Repository
    from htwutil.runner import Runner
    from asr.database import connect

    repo = Repository.find(Path.cwd())

    if select:
        print(f'Selecting {select}')

    # Defaults for functionality not currently accessible:
    chunks = 1
    sort = None
    copy = False
    patterns = '*'
    update_tree = False

    if sort:
        print(f'Sorting after {sort}')

    assert Path(database).exists(), f'file: {database} does not exist'

    db = connect(database)
    rows = list(db.select(select, sort=sort))

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

    for atoms, directory in make_folder_tree(
            repo=repo, folders=folders, chunks=chunks, copy=copy,
            patterns=patterns,
            update_tree=update_tree):

        rn = Runner(repo.cache, tasks={'structure': structure},
                    directory=directory)

        # The name "structure" should likely be a choice
        future = rn.task('structure', atoms=atoms)

        # XXX need better interface, WIP in htw-util
        entry = future._entry
        entry.dump_output(atoms)
