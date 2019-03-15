def get_parameters():
    from pathlib import Path
    import json
    if Path('params.json').is_file():
        params = json.load(open('params.json', 'r'))
    else:
        params = {}

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
