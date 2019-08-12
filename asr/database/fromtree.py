from asr.utils import command, option, argument, chdir


def collect(db, level, only_include=None):
    import traceback
    from pathlib import Path
    from ase.io import read
    from asr.utils import get_recipes, get_dep_tree

    kvp = {}
    data = {}
    key_descriptions = {}
    errors = []

    atoms = read('structure.json')
    folder = str(Path().cwd())
    if only_include:
        recipes = get_dep_tree(only_include)
    else:
        recipes = get_recipes()

    for recipe in recipes:

        if not recipe.done:
            continue
        print(f'Collecting {recipe.name}')
        try:
            tmpkvp, tmpkd, tmpdata = recipe.collect()
            if tmpkvp or tmpkd or tmpdata:

                kvp.update(tmpkvp)
                data.update(tmpdata)
                key_descriptions.update(tmpkd)
        except KeyboardInterrupt:
            raise
        except Exception as x:
            error = '{}: {}'.format(recipe.name, x)
            tb = traceback.format_exc()
            errors.append((folder, error, tb))
    if db is not None:
        if level > 1:
            db.write(atoms, data=data, **kvp)
        elif level > 0:
            db.write(atoms, **kvp)
        else:
            db.write(atoms)
        metadata = db.metadata
        metadata.update({'key_descriptions': key_descriptions})
        db.metadata = metadata
    return errors


@command('asr.database.fromtree',
         add_skip_opt=False)
@argument('folders', nargs=-1)
@option('--recipe', help='Only collect data relevant for this recipe')
@option('--level', type=int,
        help=('0: Collect only atoms. '
              '1: Collect atoms+KVP. '
              '2: Collect atoms+kvp+data'))
@option('--data/--nodata',
        help='Also add data objects to database')
@option('--raiseexc', is_flag=True)
def main(folders, recipe=None, level=2, data=True, raiseexc=False):
    """Collect data from folder tree into database."""
    import os
    import traceback
    from ase.db import connect
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), 'database.db')
    db = connect(dbname)

    if not folders:
        folders = ['.']
    
    errors = []
    for i, folder in enumerate(folders):
        if not os.path.isdir(folder):
            continue
        with chdir(folder):
            print(folder, end=':\n')
            try:
                errors2 = collect(db, level=level,
                                  only_include=recipe)
            except KeyboardInterrupt:
                break
            except Exception as x:
                error = '{}: {}'.format(x.__class__.__name__, x)
                tb = traceback.format_exc()
                errors.append((folder, error, tb))
                if raiseexc:
                    raise x
            else:
                errors.extend(errors2)

    if errors:
        print('Errors:')
        for error in errors:
            print('{}\n{}: {}\n{}'.format('=' * 77, *error))


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
