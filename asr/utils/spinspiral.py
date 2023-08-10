from asr.utils.magnetism import magnetic_atoms
import numpy as np


def get_magmoms(atoms):
    from ase.calculators.calculator import PropertyNotImplementedError
    try:
        moments = atoms.get_magnetic_moments()
    except (PropertyNotImplementedError, RuntimeError):
        if atoms.has('initial_magmoms'):
            moments = atoms.get_initial_magnetic_moments()
        else:
            # Recipe default magnetic moments
            moments = magnetic_atoms(atoms)
    return moments


def true_magnetic_atoms(atoms):
    arg = magnetic_atoms(atoms)
    moments = get_magmoms(atoms)

    # Ignore ligand atoms
    magmoms = abs(moments[arg])
    try:
        primary_magmom = magmoms[np.argmax(magmoms)]
    except ValueError:
        print(f'Skipping {atoms.symbols}, no d-orbital present')

    # Percentage of primary magnetic moment
    pmom = magmoms / primary_magmom

    # Threshold on small moments
    is_true_mag = pmom > 0.1
    arg[arg == 1] = is_true_mag
    return arg


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def calc_tanEq(q_v, R_iv):
    return -np.angle(np.sum(np.exp(-1j * np.dot(R_iv, q_v))))


def nn(atoms, ref: int = 0):
    '''
    Output:
    R_a: np.array with lattice vectors to nearest neighbours
    npos_av: np.array with coordinates of nearest neighbours
    '''
    # Select reference atom
    pos_av = atoms.get_positions()[magnetic_atoms(atoms)]
    cell_cv = atoms.get_cell()

    allpos_av = []
    R_a = []
    for pos_v in pos_av:
        for n in [-1, 0, 1]:
            for m in [-1, 0, 1]:
                allpos_av.append(pos_v + n * cell_cv[0] + m * cell_cv[1])
                R_a.append(n * cell_cv[0] + m * cell_cv[1])

    allpos_av = np.asarray(allpos_av)
    R_a = np.asarray(R_a)

    R_nn = np.linalg.norm(allpos_av - pos_av[ref], axis=1)
    r = np.partition(R_nn, 2)[2]
    R_a = R_a[np.isclose(R_nn, r)]
    npos_av = allpos_av[np.isclose(R_nn, r)]
    return R_a, npos_av


def get_afms(atoms):
    """
    Takes an atoms object
    and creates a list of non-equivalent antiferromagnetic structures.
    Note can only take even number of moment such that the antiferromagnetic
    condition is sum(moments) = 0
    Returns [[0, 1, -1, 0, 0, ...]]
    """

    import numpy as np

    moments = get_magmoms(atoms)
    arg = true_magnetic_atoms(atoms)

    # Construct list with #up and downs
    start = np.ones(sum(arg))
    start[:len(start) // 2] = -1
    # If number of magnetic atoms is odd, cannot generate afms
    if sum(start) != 0:
        return []

    # Create all collinear afm structures
    from itertools import permutations
    afms = np.array(list(set(permutations(start))))

    # Remove inversion symmetric structures
    for i in range(len(afms) // 2):
        mask = np.all(-afms[i] == afms, axis=1)
        afms = afms[np.invert(mask), :]

    # Apply the afm structure to the magmom list
    afm_comps = np.ones((len(afms), len(moments)))
    for n in range(len(afms)):
        afm_comps[n, :] = moments
        afm_comps[n, arg] = moments[arg] * afms[n]

    return afm_comps


def get_noncollinear_magmoms(atoms):
    fm = get_magmoms(atoms)
    afms = get_afms(atoms)
    magmoms = np.zeros((len(afms) + 1, len(atoms), 3))
    magmoms[0, :, 0] = fm
    for n in range(1, len(afms) + 1):
        magmoms[n, :, 0] = afms[n - 1]
    return magmoms


def spinspiral(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'experimental': {'soc': False},
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gsq.txt',
        'charge': 0},
        write_gpw: bool = True,
        return_calc: bool = False) -> dict:
    """Calculate the groundstate of a given spin spiral vector q_c"""
    from ase.io import read
    from ase.dft.kpoints import kpoint_convert
    from ase.dft.bandgap import bandgap
    from os import path
    atoms = read('structure.json')
    q_c = calculator['mode']['qspiral']  # spiral vector must be provided
    try:
        gpwfile = calculator['txt'].replace('.txt', '.gpw')
        restart = path.isfile(gpwfile)
    except KeyError:
        gpwfile = 'gsq.gpw'
        restart = False

    try:
        magmoms = calculator["experimental"]["magmoms"]
    except KeyError:
        magmomx = get_magmoms(atoms)
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 0] = magmomx

    R_iv, _ = nn(atoms, ref=0)
    q_v = kpoint_convert(atoms.get_cell(), skpts_kc=[q_c])[0]
    xi = calc_tanEq(q_v, R_iv)  # Arctan Equation
    angles = [xi, 0]
    moments_av = magmoms[true_magnetic_atoms(atoms)]
    for i, (mom_v, angle) in enumerate(zip(moments_av, angles)):
        moments_av[i] = rotation_matrix([0, 0, 1], angle) @ mom_v
    magmoms[true_magnetic_atoms(atoms)] = moments_av
    calculator['experimental']['magmoms'] = magmoms

    # Mandatory spin spiral parameters
    assert calculator["xc"] == 'LDA'
    assert not calculator["experimental"]['soc']
    assert calculator["symmetry"] == 'off'
    assert calculator["parallel"]['domain'] == 1
    assert calculator["parallel"]['band'] == 1

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    if restart:
        calc = get_calculator_class(name)(gpwfile)
    else:
        calc = get_calculator_class(name)(**calculator)

    # atoms.center(vacuum=4.0, axis=2)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    totmom_v, magmom_av = calc.density.estimate_magnetic_moments()
    gap, k1, k2 = bandgap(calc)

    if write_gpw and not restart:
        atoms.calc.write(gpwfile)
    if return_calc:
        return calc
    else:
        return {'energy': energy, 'totmom_v': totmom_v,
                'magmom_av': magmom_av, 'gap': gap}


if __name__ == '__main__':
    spinspiral.cli()
