from asr.utils import command, option, argument, chdir


def collect(db, verbose=False, skip_forces=False, references=None):
    import traceback
    from pathlib import Path
    from ase.io import read
    from asr.utils import get_recipes

    kvp = {}
    data = {}
    key_descriptions = {}
    errors = []

    atoms = read('structure.json')
    folder = str(Path().cwd())
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
        db.write(atoms, data=data, **kvp)
        metadata = db.metadata
        metadata.update({'key_descriptions': key_descriptions})
        db.metadata = metadata
    return errors


@command('asr.collect')
@argument('folders', nargs=-1)
@option('--references', default=None, type=str, help='Reference phases')
@option('--verbose', default=False)
@option('--skipforces', default=False)
def main(folders, references, verbose, skipforces):
    """Collect data in ase database"""
    import os
    import traceback
    from pathlib import Path
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
                if references:
                    references = Path(references).resolve()
                errors2 = collect(
                    db,
                    verbose=verbose,
                    skip_forces=skipforces,
                    references=references)
            except KeyboardInterrupt:
                break
            except Exception as x:
                error = '{}: {}'.format(x.__class__.__name__, x)
                tb = traceback.format_exc()
                errors.append((folder, error, tb))
            else:
                errors.extend(errors2)

    if errors:
        print('Errors:')
        for error in errors:
            print('{}\n{}: {}\n{}'.format('=' * 77, *error))


group = 'postprocessing'

if __name__ == '__main__':
    main()
