import click

from functools import partial
click.option = partial(click.option, show_default=True)


def get_recipes():
    import importlib
    from pathlib import Path

    files = Path(__file__).parent.parent.glob('[a-zA-Z]*.py')
    recipes = []
    for file in files:
        name = file.with_suffix('').name
        module = importlib.import_module(f'asr.{name}')
        recipes.append(module)
    return recipes


def get_parameters(key=None):
    from pathlib import Path
    import json
    if Path('params.json').is_file():
        with open('params.json', 'r') as fd:
            params = json.load(fd)
    else:
        params = {}

    if key and key in params:
        params = params[key]

    return params


def is_magnetic():
    import numpy as np
    atoms = get_start_atoms()
    magmom_a = atoms.get_initial_magnetic_moments()
    maxmom = np.max(np.abs(magmom_a))
    if maxmom > 1e-3:
        return True
    else:
        return False


def get_dimensionality():
    import numpy as np
    start = get_start_atoms()
    nd = int(np.sum(start.get_pbc()))
    return nd


mag_elements = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'}


def magnetic_atoms(atoms):
    import numpy as np
    return np.array([symbol in mag_elements
                     for symbol in atoms.get_chemical_symbols()],
                    dtype=bool)


def update_defaults(key, params={}):
    params.update(get_parameters(key))

    def update_defaults_dec(func):
        fparams = func.__click_params__
        for param in fparams:
            for externaldefault in params:
                if externaldefault == param.name:
                    param.default = params[param.name]
                    break

        return func
    return update_defaults_dec


def get_start_file():
    "Get starting atomic structure"
    from pathlib import Path
    fnames = list(Path('.').glob('start.*'))
    assert len(fnames) == 1, fnames
    return str(fnames[0])


def get_start_atoms():
    from ase.io import read
    fname = get_start_file()
    atoms = read(str(fname))
    return atoms


def get_start_parameters(atomfile=None):
    import json
    if atomfile is None:
        try:
            atomfile = get_start_file()
        except AssertionError:
            return {}
    with open(atomfile, 'r') as fd:
        asejsondb = json.load(fd)
    params = asejsondb.get('1').get('calculator_parameters', {})

    return params
