"""Functionality for converting a folder tree to an ASE database."""

from typing import Union, List
from asr.core import command, option, argument, chdir, read_json
from asr.database.key_descriptions import key_descriptions as asr_kd
from pathlib import Path
import glob


def parse_kd(key_descriptions):
    import re

    tmpkd = {}

    for key, desc in key_descriptions.items():
        descdict = {'type': None,
                    'iskvp': False,
                    'shortdesc': '',
                    'longdesc': '',
                    'units': ''}
        if isinstance(desc, dict):
            descdict.update(desc)
            tmpkd[key] = desc
            continue

        assert isinstance(desc, str), \
            f'Key description has to be dict or str. ({desc})'
        # Get key type
        desc, *keytype = desc.split('->')
        if keytype:
            descdict['type'] = keytype

        # Is this a kvp?
        iskvp = desc.startswith('KVP:')
        descdict['iskvp'] = iskvp
        desc = desc.replace('KVP:', '').strip()

        # Find units
        m = re.search(r"\[(.*)\]", desc)
        unit = m.group(1) if m else ''
        if unit:
            descdict['units'] = unit
        desc = desc.replace(f'[{unit}]', '').strip()

        # Find short description
        m = re.search(r"\!(.*)\!", desc)
        shortdesc = m.group(1) if m else ''
        if shortdesc:
            descdict['shortdesc'] = shortdesc

        # Everything remaining is the long description
        longdesc = desc.replace(f'!{shortdesc}!', '').strip()
        if longdesc:
            descdict['longdesc'] = longdesc
            if not shortdesc:
                descdict['shortdesc'] = descdict['longdesc']
        tmpkd[key] = descdict

    return tmpkd


tmpkd = parse_kd({key: value
                  for dct in asr_kd.values()
                  for key, value in dct.items()})


def get_key_value_pairs(resultsdct):
    """Extract key-value-pairs from results dictionary."""
    kvp = {}
    for key, desc in tmpkd.items():
        if (key in resultsdct and desc['iskvp']
           and resultsdct[key] is not None):
            kvp[key] = resultsdct[key]

    return kvp


def collect_file(filename: str):
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
    from asr.core import read_json
    data = {}
    results = read_json(filename)
    data[filename] = results

    # Find and try to collect related files for this resultsfile
    files = results.get('__files__', {})
    extra_files = results.get('__requires__', {}).copy()
    extra_files.update(results.get('__creates__', {}))

    for extrafile, checksum in extra_files.items():
        assert extrafile not in data, f'{extrafile} already collected!'

        if extrafile in files:
            continue
        file = Path(extrafile)

        if not file.is_file():
            print(f'Warning: Required file {file.absolute()}'
                  ' doesn\'t exist.')
            continue

        if file.suffix == '.json':
            dct = read_json(extrafile)
        else:
            dct = {'pointer': str(file.absolute())}
        data[extrafile] = dct

    links = results.get('__links__', {})
    kvp = get_key_value_pairs(results)
    return kvp, data, links


def collect_folder(folder: Path, atomsname: str, patterns: List[str]):
    """Collect data from a material folder."""
    from ase.io import read
    from ase.parallel import world
    from asr.database.material_fingerprint import main as mf
    from fnmatch import fnmatch

    with chdir(folder):
        kvp = {'folder': str(folder)}
        data = {'__links__': {}}

        if not Path(atomsname).is_file():
            return None, None, None

        if world.size == 1:
            if not mf.done:
                mf()

        atoms = read(atomsname, parallel=False)
        data[atomsname] = read_json(atomsname)
        for filename in glob.glob('*'):
            for pattern in patterns:
                if fnmatch(filename, pattern):
                    break
            else:
                continue
            tmpkvp, tmpkd, tmpdata, tmplinks = \
                collect_file(str(filename))
            if tmpkvp or tmpkd or tmpdata or tmplinks:
                kvp.update(tmpkvp)
                data.update(tmpdata)
                data['__links__'].update(tmplinks)
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


@command('asr.database.fromtree')
@argument('folders', nargs=-1, type=str)
@option('--patterns', help='Only select files matching pattern.', type=str)
@option('--dbname', help='Database name.', type=str)
@option('-m', '--metadata-from-file', help='Get metadata from file.',
        type=str)
def main(folders: Union[str, None] = None,
         patterns: str = 'info.json,results-asr.*.json',
         dbname: str = 'database.db', metadata_from_file: str = None):
    """Collect ASR data from folder tree into an ASE database."""
    from ase.db import connect
    from ase.parallel import world

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

    folders.sort()
    patterns = patterns.split(',')
    # We use absolute path because of chdir below!
    dbpath = Path(dbname).absolute()
    metadata = {}
    if metadata_from_file:
        metadata.update(read_json(metadata_from_file))

    if world.size > 1:
        mydbname = dbpath.parent / f'{dbname}.{world.rank}.db'
        myfolders = folders[world.rank::world.size]
    else:
        mydbname = str(dbpath)
        myfolders = folders

    nfolders = len(myfolders)
    keys = set()
    with connect(mydbname, serial=True) as db:
        for ifol, folder in enumerate(myfolders):
            if world.size > 1:
                print(f'Collecting folder {folder} on rank {world.rank} '
                      f'({ifol + 1}/{nfolders})',
                      flush=True)
            else:
                print(f'Collecting folder {folder} ({ifol}/{nfolders})',
                      flush=True)
            atoms, key_value_pairs, data = collect_folder(folder,
                                                          atomsname,
                                                          patterns)

            identifier_kvp = make_data_identifiers(data.keys())
            key_value_pairs.update(identifier_kvp)
            keys.update(key_value_pairs.keys())
            db.write(atoms, data=data, **key_value_pairs)

    metadata['keys'] = sorted(list(keys))
    db.metadata = metadata

    if world.size > 1:
        # Then we have to collect the separately collected databases
        # to a single final database file.
        world.barrier()
        if world.rank == 0:
            print(f'Merging separate database files to {dbname}',
                  flush=True)
            nmat = 0
            keys = set()
            with connect(dbname, serial=True) as db2:
                for rank in range(world.size):
                    dbrankname = f'{dbname}.{rank}.db'
                    print(f'Merging {dbrankname} into {dbname}', flush=True)
                    with connect(f'{dbrankname}', serial=True) as db:
                        for row in db.select():
                            kvp = row.get('key_value_pairs', {})
                            db2.write(row, data=row.get('data'), **kvp)
                            nmat += 1
                    keys.update(set(db.metadata['keys']))

            print('Done. Setting metadata.', flush=True)
            metadata['keys'] = sorted(list(keys))
            db2.metadata = metadata
            nmatdb = len(db2)
            assert nmatdb == nmat, \
                ('Merging of databases went wrong, '
                 f'number of materials changed: {nmatdb} != {nmat}')


if __name__ == '__main__':
    main.cli()
