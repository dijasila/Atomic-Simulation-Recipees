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
    
