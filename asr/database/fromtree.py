from asr.utils import command, option, argument, chdir


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
    from asr.utils import get_recipes, get_dep_tree
    import glob
    from pathlib import Path
    from asr.utils import md5sum

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
                    # print(folder, end=':\n')
                    kvp = {}
                    data = {}
                    key_descriptions = {}

                    if not Path(atomsname).is_file():
                        # print(f'{folder} doesn\'t contain '
                        #       f'{atomsname}. Skipping.')
                        continue

                    # The atomic structure uniquely defines the folder
                    kvp['asr_id'] = md5sum(atomsname)
                    atoms = read(atomsname)
                    if selectrecipe:
                        recipes = get_dep_tree(selectrecipe)
                    else:
                        recipes = get_recipes()

                    for recipe in recipes:
                        if not recipe.done:
                            continue
                        # print(f'Collecting {recipe.name}')
                        tmpkvp, tmpkd, tmpdata = recipe.collect()
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
