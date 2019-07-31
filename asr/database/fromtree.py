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
        recipes = get_recipes(sort=True)

    for recipe in recipes:
        try:
            tmpkvp, tmpkd, tmpdata = recipe.collect(atoms=atoms)
            if tmpkvp or tmpkd or tmpdata:
                print(f'Collecting {recipe.name}')
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
@option('--recipe', default=None,
        help='Only collect data relevant for this recipe')
@option('--level', type=int, default=2,
        help=('0: Collect only atoms. '
              '1: Collect atoms+KVP. '
              '2: Collect atoms+kvp+data'))
@option('--data/--nodata', default=True,
        help='Also add data objects to database')
@option('--raiseexc', is_flag=True, default=False)
def main(folders, recipe, level, raiseexc):
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


if __name__ == '__main__':
    main()
