from asr.core import command, option, argument, chdir


def get_kvp_kd(resultdct):
    import re
    kvp = {}
    key_descriptions = {}

    if '__key_descriptions__' not in resultdct:
        return {}, {}

    tmpkd = {}

    for key, desc in resultdct['__key_descriptions__'].items():
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
            'Key description has to be dict or str.'
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
        m = re.search(r"\((.*)\)", desc)
        shortdesc = m.group(1) if m else ''
        if shortdesc:
            descdict['shortdesc'] = shortdesc

        # Everything remaining is the long description
        longdesc = desc.replace(f'({shortdesc})', '').strip()
        if longdesc:
            descdict['longdesc'] = longdesc
            if not shortdesc:
                descdict['shortdesc'] = descdict['longdesc']
        tmpkd[key] = descdict

    for key, desc in tmpkd.items():
        key_descriptions[key] = \
            (desc['shortdesc'], desc['longdesc'], desc['units'])

        if key in resultdct and desc['iskvp']:
            kvp[key] = resultdct[key]

    return kvp, key_descriptions


def collect(filename):
    from pathlib import Path
    from asr.core import read_json, md5sum
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
            print(f'Warning: Required file {extrafile}'
                  ' doesn\'t exist.')
            continue

        if file.suffix == '.json':
            dct = read_json(extrafile)
        else:
            dct = {'pointer': str(file.absolute()),
                   '__md5__': md5sum(extrafile)}
        data[extrafile] = dct

    links = results.get('__links__', {})
    # Parse key descriptions to get long,
    # short, units and key value pairs
    kvp, key_descriptions = get_kvp_kd(results)
    return kvp, key_descriptions, data, links


@command('asr.database.fromtree')
@argument('folders', nargs=-1)
@option('--patterns', help='Only select files matching pattern.')
@option('--dbname', help='Database name.')
@option('-m', '--metadata-from-file', help='Get metadata from file.')
def main(folders, patterns='info.json,results-asr.*.json',
         dbname='database.db', metadata_from_file=None):
    """Collect ASR data from folder tree into an ASE database."""
    import os
    from ase.db import connect
    from ase.io import read
    from asr.core import read_json
    import glob
    from pathlib import Path
    from asr.database.material_fingerprint import main as mat_finger
    from fnmatch import fnmatch
    from click import progressbar

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

    patterns = patterns.split(',')

    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), dbname)
    metadata = {'key_descriptions': {}}
    if metadata_from_file:
        metadata.update(read_json(metadata_from_file))

    with connect(dbname) as db:
        with progressbar(folders, label=f'Collecting to {dbname}',
                         item_show_func=item_show_func) as bar:
            for folder in bar:
                with chdir(folder):
                    kvp = {}
                    data = {'__links__': {}}
                    key_descriptions = {}

                    if not mat_finger.done:
                        mat_finger(silence=True)

                    if not Path(atomsname).is_file():
                        continue

                    atoms = read(atomsname)
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
                            key_descriptions.update(tmpkd)
                            data['__links__'].update(tmplinks)

                    db.write(atoms, data=data, **kvp)
                    metadata['key_descriptions'].update(key_descriptions)
    db.metadata = metadata


tests = [
    {'cli': ['asr run setup.materials',
             'asr run database.totree materials.json --run'
             ' --atomsname structure.json',
             'asr run database.fromtree tree/*/*/*/',
             'asr run database.totree database.db '
             '-t newtree/{formula} --run'],
     'results': [{'file': 'newtree/Ag/unrelaxed.json'}]},
    {'cli': ['asr run setup.materials',
             'asr run database.totree materials.json -s "natoms\\<2" --run'
             ' -t tree/{formula} --atomsname structure.json',
             'asr run structureinfo in tree/*/',
             'asr run database.fromtree tree/*/',
             'asr run database.totree database.db '
             '-t newtree/{formula} --run --data'],
     'results': [{'file': 'newtree/Ag/unrelaxed.json'},
                 {'file': 'newtree/Ag/results-asr-structureinfo@main.json'}]}
]


if __name__ == '__main__':
    main.cli()
