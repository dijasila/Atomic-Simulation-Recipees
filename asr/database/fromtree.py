from asr.core import command, option, argument, chdir


def collect(recipe):
    import re
    from pathlib import Path
    from asr.core import read_json, md5sum
    kvp = {}
    key_descriptions = {}
    data = {}

    resultfile = f'results-{recipe.name}.json'
    results = read_json(resultfile)
    msg = f'{recipe.name}: {resultfile} already in data'
    assert resultfile not in data, msg
    data[resultfile] = results

    extra_files = results.get('__requires__', {})
    extra_files.update(results.get('__creates__', {}))
    if extra_files:
        for filename, checksum in extra_files.items():
            if filename in data:
                continue
            file = Path(filename)
            if not file.is_file():
                print(f'Warning: Required file {filename}'
                      ' doesn\'t exist')

            filetype = file.suffix
            if filetype == '.json':
                dct = read_json(filename)
            elif recipe.todict and filetype in recipe.todict:
                dct = recipe.todict[filetype](filename)
                dct['__md5__'] = md5sum(filename)
            else:
                dct = {'pointer': str(file.absolute()),
                       '__md5__': md5sum(filename)}
            data[filename] = dct

    # Parse key descriptions to get long,
    # short, units and key value pairs
    if '__key_descriptions__' in results:
        tmpkd = {}

        for key, desc in results['__key_descriptions__'].items():
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
            m = re.search(r"\[(\w+)\]", desc)
            unit = m.group(1) if m else ''
            if unit:
                descdict['units'] = unit
            desc = desc.replace(f'[{unit}]', '').strip()

            # Find short description
            m = re.search(r"\((\w+)\)", desc)
            shortdesc = m.group(1) if m else ''

            # The results is the long description
            longdesc = desc.replace(f'({shortdesc})', '').strip()
            if longdesc:
                descdict['longdesc'] = longdesc
            tmpkd[key] = descdict

        for key, desc in tmpkd.items():
            key_descriptions[key] = \
                (desc['shortdesc'], desc['longdesc'], desc['units'])

            if key in results and desc['iskvp']:
                kvp[key] = results[key]

    return kvp, key_descriptions, data


@command('asr.database.fromtree',
         dependencies=['asr.structureinfo'])
@argument('folders', nargs=-1)
@option('--selectrecipe', help='Only collect data relevant for this recipe')
@option('--level', type=int,
        help=('0: Collect only atoms. '
              '1: Collect atoms+KVP. '
              '2: Collect atoms+kvp+data'))
@option('--data/--nodata',
        help='Also add data objects to database')
@option('--atomsname', help='File containing atomic structure.')
def main(folders, selectrecipe=None, level=2, data=True,
         atomsname='structure.json'):
    """Collect ASR data from folder tree into an ASE database."""
    import os
    from ase.db import connect
    from ase.io import read
    from asr.core import get_recipes, get_dep_tree
    import glob
    from pathlib import Path

    if not folders:
        folders = ['.']
    else:
        tmpfolders = []
        for folder in folders:
            tmpfolders.extend(glob.glob(folder))
        folders = tmpfolders

    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), 'database.db')
    from click import progressbar

    def item_show_func(item):
        return str(item)

    metadata = {}
    with connect(dbname) as db:
        with progressbar(folders, label='Collecting to database.db',
                         item_show_func=item_show_func) as bar:
            for folder in bar:
                with chdir(folder):
                    kvp = {}
                    data = {}
                    key_descriptions = {}

                    if not Path(atomsname).is_file():
                        continue

                    # The atomic structure uniquely defines the folder
                    atoms = read(atomsname)
                    if selectrecipe:
                        recipes = get_dep_tree(selectrecipe)
                    else:
                        recipes = get_recipes()

                    for recipe in recipes:
                        if not recipe.done:
                            continue

                        tmpkvp, tmpkd, tmpdata = collect(recipe)
                        if tmpkvp or tmpkd or tmpdata:
                            kvp.update(tmpkvp)
                            data.update(tmpdata)
                            key_descriptions.update(tmpkd)

                    if level > 1:
                        db.write(atoms, data=data, **kvp)
                    elif level > 0:
                        db.write(atoms, **kvp)
                    else:
                        db.write(atoms)
                    metadata.update({'key_descriptions': key_descriptions})
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
