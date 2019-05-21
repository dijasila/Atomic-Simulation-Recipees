from asr.utils import command, option, argument, chdir


def collect(db, verbose=False, skip_forces=False, references=None):
    import traceback
    from pathlib import Path
    from importlib import import_module
    from ase.io import read

    kvp = {}
    data = {}
    key_descriptions = {}
    errors = []

    atoms = read('start.json')
    folder = str(Path().cwd())
    steps = []
    names = []
    pathlist = Path(__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        module = import_module('asr.' + name, package='')
        try:
            steps.append(module.collect_data)
            names.append(module.__name__)
        except AttributeError:
            continue

    for name, step in zip(names, steps):
        try:
            tmpkvp, tmpkd, tmpdata = step(atoms=atoms)
            if tmpkvp or tmpkd or tmpdata:
                print(f'Collecting {name}')
                kvp.update(tmpkvp)
                data.update(tmpdata)
                key_descriptions.update(tmpkd)
        except KeyboardInterrupt:
            raise
        except Exception as x:
            error = '{}: {}'.format(name, x)
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
