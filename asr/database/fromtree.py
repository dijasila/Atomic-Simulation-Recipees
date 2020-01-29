from asr.core import command, option, argument, chdir
from asr.database.key_descriptions import key_descriptions as asr_kd


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


def get_kvp_kd(resultsdct):
    # Parse key descriptions to get long,
    # short, units and key value pairs
    key_descriptions = {}
    kvp = {}
    for key, desc in tmpkd.items():
        key_descriptions[key] = \
            (desc['shortdesc'], desc['longdesc'], desc['units'])

        if (key in resultsdct and desc['iskvp'] and
            resultsdct[key] is not None):
            kvp[key] = resultsdct[key]

    return kvp, key_descriptions


def collect(filename):
    from pathlib import Path
    from asr.core import read_json
    data = {}

    # resultfile = f'results-{recipe.name}.json'
    results = read_json(filename)
    msg = f'{filename} already in data!'
    assert filename not in data, msg
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
    kvp, key_descriptions = get_kvp_kd(results)
    return kvp, key_descriptions, data, links


tests = [
    {'cli': ['asr run setup.materials',
             ('asr run "database.totree materials.json --run'
              ' --atomsname structure.json"'),
             'asr run "database.fromtree tree/*/*/*/"',
             ('asr run "database.totree database.db '
              '-t newtree/{formula} --run"')],
     'results': [{'file': 'newtree/Ag/structure.json'}]},
    {'cli': ['asr run setup.materials',
             'asr run "database.totree materials.json -s natoms<2 --run'
             ' -t tree/{formula} --atomsname structure.json"',
             'asr run structureinfo tree/*/',
             'asr run "database.fromtree tree/*/"',
             'asr run "database.totree database.db '
             '-t newtree/{formula} --run"'],
     'results': [{'file': 'newtree/Ag/structure.json'},
                 {'file': 'newtree/Ag/results-asr.structureinfo.json'}]}
]


@command('asr.database.fromtree',
         tests=tests)
@argument('folders', nargs=-1)
@option('--patterns', help='Only select files matching pattern.')
@option('--dbname', help='Database name.')
@option('-m', '--metadata-from-file', help='Get metadata from file.')
def main(folders=None, patterns='info.json,results-asr.*.json',
         dbname='database.db', metadata_from_file=None):
    """Collect ASR data from folder tree into an ASE database."""
    from ase.db import connect
    from ase.io import read
    from asr.core import read_json
    import glob
    from pathlib import Path
    from fnmatch import fnmatch
    from asr.database.material_fingerprint import main as mf
    from gpaw.mpi import world

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
            with chdir(folder):
                kvp = {'folder': str(folder),
                       'absfolder': str(Path(folder).resolve())}
                data = {'__links__': {}}

                if not Path(atomsname).is_file():
                    continue

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
                        collect(str(filename))
                    if tmpkvp or tmpkd or tmpdata or tmplinks:
                        kvp.update(tmpkvp)
                        data.update(tmpdata)
                        data['__links__'].update(tmplinks)

            keys.update(kvp.keys())
            db.write(atoms, data=data, **kvp)

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
