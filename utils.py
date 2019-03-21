import click
from functools import partial
click.option = partial(click.option, show_default=True)


def get_parameters(key=None):
    from pathlib import Path
    import json
    if Path('params.json').is_file():
        params = json.load(open('params.json', 'r'))
    else:
        params = {}

    if key and key in params:
        params = params[key]

    return params


def is_magnetic():
    import numpy as np
    from ase.io import read
    atoms = read('start.traj')
    magmom_a = atoms.get_initial_magnetic_moments()
    maxmom = np.max(np.abs(magmom_a))
    if maxmom > 1e-3:
        return True
    else:
        return False


def get_dimensionality():
    from ase.io import read
    import numpy as np
    start = read('start.traj')
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


def update_defaults(key):
    params = get_parameters(key)

    def update_defaults_dec(func):
        fparams = func.__click_params__
        for param in fparams:
            for externaldefault in params:
                if externaldefault == param.name:
                    param.default = params[param.name]
                break

        return func
    return update_defaults_dec
