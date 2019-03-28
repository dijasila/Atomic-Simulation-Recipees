import click
import os
from contextlib import contextmanager


@contextmanager
def chdir(folder):
    dir = os.getcwd()
    os.chdir(str(folder))
    yield
    os.chdir(dir)


def collect(db, verbose=False, skip_forces=False, references=None):
    import traceback
    from pathlib import Path
    from importlib import import_module
    from asr.utils import get_start_atoms

    kvp = {}
    data = {}
    key_descriptions = {}
    errors = []

    atoms = get_start_atoms()
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
            print(f'Collecting {name}')
            step(
                kvp=kvp,
                data=data,
                key_descriptions=key_descriptions,
                atoms=atoms,
                verbose=verbose)
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


@click.command()
@click.argument('folders', nargs=-1)
@click.option('--references', default=None, type=str, help='Reference phases')
@click.option('--verbose', default=False)
@click.option('--skipforces', default=False)
def main(folders, references, verbose, skipforces):
    """Collect data in ase database"""
    import os
    import traceback
    from pathlib import Path
    from ase.db import connect
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), 'database.db')
    db = connect(dbname)

    errors = []
    for i, folder in enumerate(folders):
        if not os.path.isdir(folder):
            continue
        with chdir(folder):
            print(folder, end=': ')
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
                print(error)
                errors.append((folder, error, tb))
            else:
                errors.extend(errors2)

    if errors:
        print('Errors:')
        for error in errors:
            print('{}\n{}: {}\n{}'.format('=' * 77, *error))


group = 'Postprocessing'

if __name__ == '__main__':
    main()
