import numpy as np


def is_magnetic():
    import numpy as np
    from ase.io import read
    atoms = read('structure.json')
    magmom_a = atoms.get_initial_magnetic_moments()
    maxmom = np.max(np.abs(magmom_a))
    if maxmom > 1e-3:
        return True
    else:
        return False


def get_dimensionality():
    from ase.io import read
    atoms = read('structure.json')
    nd = int(np.sum(atoms.get_pbc()))
    return nd


mag_elements = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'}


def magnetic_atoms(atoms):
    return np.array([symbol in mag_elements
                     for symbol in atoms.get_chemical_symbols()],
                    dtype=bool)


def singleprec_dict(dct):
    assert isinstance(dct, dict), f'Input {dct} is not dict.'

    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            elif value.dtype == np.complex128:
                value = value.astype(np.complex64)
            dct[key] = value
        elif isinstance(value, dict):
            dct[key] = singleprec_dict(value)

    return dct
