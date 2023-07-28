"""Convert a folder tree to an ASE database."""

from dataclasses import dataclass
import tempfile
from typing import Union, List
from ase import Atoms
from ase.io import read
from ase.db import connect
from asr.core import command, option, argument, chdir, read_json, ASRResult
from asr.database.material_fingerprint import main as mf
from asr.database.material_fingerprint import get_uid_of_atoms, \
    get_hash_of_atoms
from asr.database.check import main as check_database
import multiprocessing
from pathlib import Path
import os
import glob
import sys
import traceback


class MissingUIDS(Exception):
    pass


def get_key_value_pairs(resultsdct: dict):
    """Extract key-value-pairs from results dictionary.

    Note to determine which key in the results dictionary is a
    key-value-pair we parse the data in `asr.database.key_descriptions`.

    Parameters
    ----------
    resultsdct: dict
        Dictionary containing asr results file.

    Returns
    -------
    kvp: dict
        key-value-pairs.
    """
    from asr.database.key_descriptions import key_descriptions as asr_kd

    all_kds = {}
    for section, dct in asr_kd.items():
        all_kds.update(dct)

    kvp = {}
    for key, desc in all_kds.items():
        if (key in resultsdct and desc.iskvp and resultsdct[key] is not None):
            kvp[key] = resultsdct[key]

    return kvp


class CollectionFailed(Exception):
    pass


def collect_file(filename: Path):
    """Collect a single file.

    Parameters
    ----------
    filename: str

    Returns
    -------
    kvp: dict
        Key-value pairs
    data: dict
        Dict with keys=filenames where filenames is the input filename
        and any additional files that were created or required by this recipe.
    links: dict
        Dict with keys

    """

    try:
        return _collect_file(filename)
    except Exception as err:
        raise CollectionFailed(filename.resolve()) from err


def _collect_file(filename):
    from asr.core import read_json
    data = {}
    results = read_json(filename)
    if isinstance(results, ASRResult):
        dct = results.format_as('dict')
    else:
        dct = results

    # This line is what makes someone else define "has_asr_thing_recipe" keys:
    data[str(filename)] = dct

    kvp = get_key_value_pairs(results)
    return kvp, data


def collect_info(filename: Path):
    """Collect info.json."""
    from asr.core import read_json
    kvp = read_json(filename)
    data = {str(filename): kvp}

    return kvp, data


def collect_links_to_child_folders(folder: Path, atomsname):
    """Collect links to all subfolders.

    Parameters
    ----------
    folder: Path
        Path to folder.
    atomsname: str
        Name of file containing atoms, i.e. 'structure.json'.

    Returns
    -------
    children: dict
        Dictionary with key=relative path to child material and
        value=uid of child material, i.e.: {'strains':
        'Si2-abcdefghiklmn'}.

    """
    children = {}

    for root, dirs, files in os.walk(folder, topdown=True, followlinks=False):
        this_folder = Path(root).resolve()

        if atomsname in files:
            with chdir(this_folder):
                atoms = read(atomsname, parallel=False)
                uid = get_material_uid(atoms)
                children[root] = uid
    return children


def get_material_uid(atoms: Atoms):
    """Get UID of atoms."""
    if mf.done:
        return read_json(
            'results-asr.database.material_fingerprint.json')['uid']

    hash = get_hash_of_atoms(atoms)
    return get_uid_of_atoms(atoms, hash)


def collect_folder(
    folder: Path, atomsname: str, patterns: List[str] = [''],
    exclude_patterns: List[str] = [],
    children_patterns=[],
):
    """Collect data from a material folder.

    Parameters
    ----------
    folder: Path
        Path to folder.
    atomsname: str
        Name of file containing atoms, i.e. 'structure.json'.
    patterns: List[str]
        List of patterns marking which files to include.
    exclude_patterns: List[str]
        List of patterns to exlude, takes precedence over patterns.

    Returns
    -------
    atoms: Atoms
        Atomic structure.
    kvp: dict
        Key-value-pairs.
    data: dict
        Dictionary containing data files and links.

    """
    from fnmatch import fnmatch

    with chdir(folder.resolve()):
        if not Path(atomsname).is_file():
            return None, None, None

        atoms = read(atomsname, parallel=False)

        uid = get_material_uid(atoms)
        kvp = {'folder': str(folder),
               'uid': uid}
        data = {'__children__': {}}
        data[atomsname] = read_json(atomsname)
        for name in Path().glob('*'):
            if name.is_dir() and any(fnmatch(name, pattern)
                                     for pattern in children_patterns):
                children = collect_links_to_child_folders(name, atomsname)
                data['__children__'].update(children)
            else:
                if name.is_file() and name.name == 'info.json':
                    tmpkvp, tmpdata = collect_info(name)
                elif name.is_file() and any(fnmatch(name, pattern)
                                            for pattern in exclude_patterns):
                    continue
                elif name.is_file() and any(fnmatch(name, pattern)
                                            for pattern in patterns):
                    tmpkvp, tmpdata = collect_file(name)
                else:
                    continue

                for key, value in tmpkvp.items():
                    # Skip values not suitable for a database column:
                    if key == 'folder':
                        continue
                    if key == 'etot':
                        # Clash between etot from relax and gs!
                        # What do we do about this?
                        continue
                    if isinstance(value, (bool, int, float, str)):
                        if key in kvp and kvp[key] != value:
                            raise ValueError(
                                f'Found {key}={value} in {name}: '
                                f'{key} already read once: '
                                f'{key}={kvp[key]}')
                        kvp[key] = value

                data.update(tmpdata)

        if not data['__children__']:
            del data['__children__']

    return atoms, kvp, data


def make_data_identifiers(filenames: List[str]):
    """Make key-value-pairs for identifying data files.

    This function looks at the keys of `data` and identifies any
    result files. If a result file has been identified a key value
    pair with name has_asr_name=True will be returned. I.e. if
    results-asr.gs@calculate.json is in `data` a key-value-pair with
    name `has_asr_gs_calculate=True` will be generated

    Parameters
    ----------
    filenames: List[str]
        List of file names.

    Returns
    -------
    dict
        Dict containing identifying key-value-pairs,
        i.e. {'has_asr_gs_calculate': True}.
    """
    kvp = {}
    for key in filter(lambda x: x.startswith('results-'), filenames):
        recipe = key[8:-5].replace('.', '_').replace('@', '_')
        name = f'has_{recipe}'
        kvp[name] = True
    return kvp


def recurse_through_folders(folder, atomsname):
    """Find all folders from folder that contain atomsname."""
    folders = []
    for root, dirs, files in os.walk(folder, topdown=True, followlinks=False):
        if atomsname in files:
            folders.append(root)
    return folders


@dataclass
class RowInput:
    atoms: Atoms
    key_value_pairs: dict
    data: dict


def _collect_folders(folders: List[str],
                     atomsname: str = None,
                     patterns: List[str] = None,
                     exclude_patterns: List[str] = None,
                     children_patterns: List[str] = None,
                     dbname: str = None,
                     collection_hook=None,
                     jobid: int = None):
    """Collect `myfolders` to `mydbname`."""
    nfolders = len(folders)
    with connect(dbname, serial=True) as db:
        for ifol, folder in enumerate(folders):
            string = f'Collecting folder {folder} ({ifol + 1}/{nfolders})'
            if jobid is not None:
                print(f'Subprocess #{jobid} {string}', flush=True)
            else:
                print(string)

            atoms, key_value_pairs, data = collect_folder(
                Path(folder),
                atomsname,
                patterns,
                exclude_patterns,
                children_patterns=children_patterns)

            if atoms is None:
                continue

            identifier_kvp = make_data_identifiers(data.keys())
            key_value_pairs.update(identifier_kvp)

            # The collection hook gets to see and modify things in
            # the form of RowInput as we write to the database.
            #
            # C2DB uses this to add extra KVPs.
            rowinput = RowInput(atoms=atoms, key_value_pairs=key_value_pairs,
                                data=data)

            try:
                if collection_hook is not None:
                    collection_hook(rowinput)
                db.write(atoms, data=data, **key_value_pairs)
            except Exception:
                print(f'folder={folder}')
                print(f'atoms={atoms}')
                print(f'data={data}')
                print(f'kvp={key_value_pairs}')
                raise


def collect_folders(folders: List[str],
                    atomsname: str = None,
                    patterns: List[str] = None,
                    exclude_patterns: List[str] = None,
                    children_patterns: List[str] = None,
                    dbname: str = None,
                    collection_hook=None,
                    jobid: int = None):
    """Collect `myfolders` to `mydbname`.

    This wraps _collect_folders and handles printing exception traceback, which
    is broken using multiproces.

    """
    try:
        return _collect_folders(folders=folders, atomsname=atomsname,
                                patterns=patterns,
                                exclude_patterns=exclude_patterns,
                                children_patterns=children_patterns,
                                dbname=dbname,
                                collection_hook=collection_hook,
                                jobid=jobid)
    except Exception:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def delegate_to_njobs(njobs, dbpath, name, folders, atomsname,
                      patterns, exclude_patterns, children_patterns, dbname,
                      collection_hook, tempdir):
    print(f'Delegating database collection to {njobs} subprocesses.')
    processes = []

    for jobid in range(njobs):
        jobdbname = tempdir / f'{name}.{jobid}.db'
        proc = multiprocessing.Process(
            target=collect_folders,
            args=(folders[jobid::njobs], ),
            kwargs={
                'jobid': jobid,
                'dbname': jobdbname,
                'atomsname': atomsname,
                'patterns': patterns,
                'collection_hook': collection_hook,
                'children_patterns': children_patterns,
                'exclude_patterns': exclude_patterns,
            })
        processes.append(proc)
        proc.start()

    for jobid, proc in enumerate(processes):
        proc.join()
        assert proc.exitcode == 0, f'Error in Job #{jobid}.'

    # Then we have to collect the separately collected databases
    # to a single final database file.
    print(f'Merging separate database files to {dbname}',
          flush=True)
    nmat = 0
    with connect(dbname, serial=True) as db2:
        for jobid in range(njobs):
            jobdbname = tempdir / f'{name}.{jobid}.db'
            assert Path(jobdbname).is_file()
            print(f'Merging {jobdbname} into {dbname}', flush=True)
            with connect(f'{jobdbname}', serial=True) as db:
                for row in db.select():
                    kvp = row.get('key_value_pairs', {})
                    data = row.get('data')
                    db2.write(row.toatoms(), data=data, **kvp)
                    nmat += 1
    print('Done.', flush=True)
    nmatdb = len(db2)
    assert nmatdb == nmat, \
        ('Merging of databases went wrong, '
         f'number of materials changed: {nmatdb} != {nmat}')

    for name in Path().glob(f'{dbname}.*.db'):
        name.unlink()


@command('asr.database.fromtree', save_results_file=False)
@argument('folders', nargs=-1, type=str)
@option('-r', '--recursive', is_flag=True,
        help='Recurse and collect subdirectories.')
@option('--children-patterns', type=str)
@option('--patterns', help='Only select files matching pattern.', type=str)
@option(
    '--exclude-patterns',
    help='Comma separated list of patterns to exclude.'
    ' Takes precedence over --patterns.',
    type=str,
)
@option('--dbname', help='Database name.', type=str)
@option('--collection-hook',
        help='<cannot be set using CLI>')
# An assertion elsewhere obligates us to expose a CLI interface to the hook,
# even though it requires python.
@option('--njobs', type=int,
        help='Delegate collection of database to NJOBS subprocesses. '
        'Can significantly speed up database collection.')
def main(folders: Union[str, None] = None,
         recursive: bool = False,
         children_patterns: str = '*',
         patterns: str = 'info.json,links.json,params.json,results-asr.*.json',
         exclude_patterns: str = '',
         dbname: str = 'database.db',
         collection_hook=None,
         njobs: int = 1) -> ASRResult:
    """Collect ASR data from folder tree into an ASE database."""
    from asr.database.key_descriptions import main as set_key_descriptions

    def item_show_func(item):
        return str(item)

    atomsname = 'structure.json'
    if not folders:
        folders = ['.']
    else:
        tmpfolders = []
        for folder in folders:
            tmpfolders.extend(glob.glob(folder))
        folders = tmpfolders

    if recursive:
        print('Recursing through folder tree...')
        newfolders = []
        for folder in folders:
            newfolders += recurse_through_folders(folder, atomsname)
        folders = newfolders
        print('Done.')

    folders.sort()
    patterns = patterns.split(',')
    exclude_patterns = exclude_patterns.split(',')
    children_patterns = children_patterns.split(',')

    # We use absolute path because of chdir in collect_folder()!
    dbpath = Path(dbname).absolute()
    name = dbpath.name

    # Delegate collection of database to subprocesses to reduce I/O time.
    if njobs > 1:
        with tempfile.TemporaryDirectory(dir='.') as tempdir:
            tempdir = Path(tempdir)
            delegate_to_njobs(
                njobs=njobs,
                dbpath=dbpath, name=name,
                folders=folders,
                atomsname=atomsname,
                patterns=patterns,
                exclude_patterns=exclude_patterns,
                children_patterns=children_patterns,
                dbname=dbname,
                tempdir=tempdir,
                collection_hook=collection_hook)
    else:
        _collect_folders(folders,
                         jobid=None,
                         dbname=dbname,
                         atomsname=atomsname,
                         patterns=patterns,
                         collection_hook=collection_hook,
                         exclude_patterns=exclude_patterns,
                         children_patterns=children_patterns)

    set_key_descriptions(dbname)
    results = check_database(dbname)
    missing_child_uids = results['missing_child_uids']
    duplicate_uids = results['duplicate_uids']

    if missing_child_uids:
        raise MissingUIDS(
            'Missing child uids in collected database. '
            'Did you collect all subfolders?')

    if duplicate_uids:
        raise MissingUIDS(
            'Duplicate uids in database.')


if __name__ == '__main__':
    main.cli()
