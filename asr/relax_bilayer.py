from asr.core import command, option, AtomsFile, DictStr
import numpy as np
from ase import Atoms
import os
from asr.utils.bilayerutils import translation


def get_energy(base, top, h, t_c, settings, callback, memo):
    from ase.calculators.dftd3 import DFTD3
    from gpaw import GPAW, PW

    try:
        h0, e0 = next(t for t in memo if np.allclose(t[0], h))
        return e0
    except StopIteration:
        pass

    calc = GPAW(mode=PW(800),
                xc=settings['xc'],
                kpts=settings['kpts'],
                symmetry={'symmorphic': False},
                occupations={'name': 'fermi-dirac',
                             'width': 0.05},
                poissonsolver={"dipolelayer": "xy"},
                charge=0,
                txt='relax_bilayer.txt')
    if settings['d3']:
        calc = DFTD3(dft=calc, cutoff=60)

    tx, ty = t_c[0], t_c[1]
    atoms = translation(tx, ty, h[0], top, base)
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()

    callback(h[0], e)

    return e


def initial_displacement(atoms, distance):
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])

    return distance + (maxz - minz)


@command('asr.relax_bilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-s', '--settings', help='Relaxation settings',
        type=DictStr())
@option('-n', '--name', help='Name of final structure file',
        default='structure.json')
@option('--tol', help='Convergence threshold',
        type=float, default=1e-2)
@option('-d', '--distance', help='Initial Distance',
        type=float, default=3)
@option('-v', '--vacuum', help='Extra vacuum',
        type=float, default=6)
def main(atoms: Atoms,
         settings: dict = {'d3': True,
                           'xc': 'PBE',
                           'mode': 'interlayer',
                           'PWE': 800,
                           'kpts': {'density': 6.0, 'gamma': True}},
         name='structure.json',
         tol=1e-2,
         distance=5,
         vacuum=6):
    from asr.core import read_json
    from ase.io import read
    from gpaw import mpi
    import scipy.optimize as sciop
    from asr.stack_bilayer import translation
    top_layer = read('toplayer.json')
    if not np.allclose(top_layer.cell, atoms.cell):
        top_layer.cell = atoms.cell.copy()
        top_layer.center()
        top_layer.write("corrected.json")

    # assert(np.allclose(top_layer.cell, atoms.cell))
    t_c = np.array(read_json('translation.json')['translation_vector']).astype(float)

    d0 = initial_displacement(atoms, distance)
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])
    w = maxz - minz
    atoms.cell[2, 2] += vacuum + w
    top_layer.cell = atoms.cell

    start_structure = translation(t_c[0], t_c[1], d0, top_layer,
                                  atoms)
    start_structure.write("startstructure.json")

    if os.path.exists('energy_curve.npy'):
        energy_curve = np.load('energy_curve.npy', allow_pickle=True)
        energy_curve = [(d, e) for d, e in energy_curve]
    else:
        energy_curve = []

    def callback_fn(h, energy):
        energy_curve.append((h, energy))
        if mpi.rank == 0:
            np.save('energy_curve.npy', np.array(energy_curve))

    def energy_fn(h):
        return get_energy(atoms, top_layer,
                          h, t_c, settings,
                          callback_fn, energy_curve)

    opt_result = sciop.minimize(energy_fn, x0=d0, method="Nelder-Mead",
                                tol=tol)
    if mpi.rank == 0:
        np.save('energy_curve.npy', np.array(energy_curve))

    if not opt_result.success:
        raise ValueError('Relaxation failed')

    hmin = opt_result.x[0]

    final_atoms = translation(t_c[0], t_c[1], hmin,
                              top_layer, atoms)

    final_atoms.write("structure.json")

    curve = np.array(energy_curve)

    P = bilayer_stiffness(curve)

    results = {'heights': curve[:, 0],
               'energies': curve[:, 1],
               'optimal_height': hmin,
               'energy': opt_result.fun,
               'curvature': P[2],
               'FittingParams': P}

    return results


def webpanel():
    """Return a webpanel showing the binding energy curve.

    Also show the fit made to determine the stiffness.
    """
    raise NotImplementedError()


def bilayer_stiffness(energy_curve):
    """Calculate bilayer stiffness.

    We define the bilayer stiffness as the curvature
    of the binding energy curve at the minimum.
    That is, we are calculating an effective spring
    constant.

    For now we include this property calculation here
    since we can get it almost for free.
    """
    # We do a second order fit using points within
    # 0.01 eV of minimum
    ds = energy_curve[:, 0]
    es = energy_curve[:, 1]
    mine = np.min(es)
    window = 0.01

    X = ds[np.abs(es - mine) < window]
    Y = es[np.abs(es - mine) < window]

    I = np.array([X**0, X**1, X**2]).T

    P, residuals, rank, singulars = np.linalg.lstsq(I, Y, rcond=None)

    return P


if __name__ == '__main__':
    main.cli()
