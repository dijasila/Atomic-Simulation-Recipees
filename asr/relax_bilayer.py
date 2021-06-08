from asr.core import command, option, AtomsFile, DictStr, prepare_result, ASRResult
import numpy as np
from ase import Atoms
import os
from asr.utils.bilayerutils import translation


def calc_setup(settings):
    from gpaw import MixerDif, Mixer, PW, GPAW
    from ase.calculators.dftd3 import DFTD3

    calcsettings = {}
    d3 = settings.pop('d3', True)
    _ = settings.pop('mode', None)  # Mode not currently used. Here for back-comp

    mixersettings = settings.pop('mixer', None)
    if mixersettings == 'mixerdif':
        # mixersettings = {'type': 'mixerdif',
        #                  'beta': 0.015, 'nmaxold': 5,
        #                  'weight': 75}
        mixersettings = {'type': 'mixerdif'}

    if type(mixersettings) != dict:
        mixersettings = {'type': 'default',
                         'beta': None, 'nmaxold': None,
                         'weight': None}

    mixertype = mixersettings.pop('type', 'default')
    if mixertype == 'mixer':
        calcsettings['mixer'] = Mixer(**mixersettings)
    elif mixertype != 'default':
        calcsettings['mixer'] = MixerDif(**mixersettings)

    calcsettings['mode'] = PW(settings.pop('PWE'))
    calcsettings['symmetry'] = {'symmorphic': False}
    calcsettings['occupations'] = {'name': 'fermi-dirac',
                                   'width': 0.05}
    calcsettings['poissonsolver'] = {'dipolelayer': 'xy'}
    calcsettings['nbands'] = '200%'
    calcsettings['txt'] = 'relax_bilayer.txt'

    calcsettings.update(settings)

    calc = GPAW(**calcsettings)
    if d3:
        calc = DFTD3(dft=calc, cutoff=60)

    return calc


def get_energy(base, top, h, t_c, calc, callback, memo):

    try:
        h0, e0 = next(t for t in memo if np.allclose(t[0], h))
        return e0
    except StopIteration:
        pass

    tx, ty = t_c[0], t_c[1]

    # tx, ty are kept fixed throughout relaxation
    # h (interlayer distance) is changed to find optimal
    # separation
    atoms = translation(tx, ty, h[0], top, base)
    atoms.calc = calc

    e = atoms.get_potential_energy()

    callback(h[0], e)

    return e


def initial_displacement(atoms, distance):
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])

    return distance + (maxz - minz)


@prepare_result
class RelaxBilayerResult(ASRResult):
    heights: np.ndarray
    energies: np.ndarray
    optimal_height: float
    energy: float
    curvature: float
    FittingParams: np.ndarray

    key_descriptions = dict(
        heights='Interlayer distance calculated during optimization',
        energies='Energies calculated during optimization',
        optimal_height='Minimum energy height',
        energy='Energy at optimal height',
        curvature='Curvature at optimal height',
        FittingParams='Parameters for second order fit')


@command('asr.relax_bilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-s', '--settings', help='Relaxation settings',
        type=DictStr())
@option('--tol', help='Convergence threshold',
        type=float)
@option('-d', '--distance', help='Initial Distance',
        type=float)
@option('-v', '--vacuum', help='Extra vacuum',
        type=float)
@option('--restart/--norestart', help='Delete memo and start relaxation from scratch',
        is_flag=True, type=bool)
@option('--outputname', help='Name of output file', type=str)
def main(atoms: Atoms,
         settings: dict = {'d3': True,
                           'xc': 'PBE',
                           'PWE': 800,
                           'maxiter': 5000,
                           'kpts': {'density': 6.0, 'gamma': True},
                           'mixer': {'type': 'default',
                                     'beta': None,
                                     'nmaxold': None,
                                     'weight': None}},
         tol: float = 1e-2,
         distance: float = 5,
         vacuum: float = 6,
         restart: bool = False,
         outputname: str = 'structure.json') -> ASRResult:
    from asr.core import read_json
    from ase.io import read
    from gpaw import mpi
    import scipy.optimize as sciop
    from asr.stack_bilayer import translation

    if restart:
        if mpi.rank == 0:
            if os.path.exists('energy_curve.npy'):
                os.remove('energy_curve.npy')
        mpi.world.barrier()

    top_layer = read('toplayer.json')
    if not np.allclose(top_layer.cell, atoms.cell):
        top_layer.cell = atoms.cell.copy()
        top_layer.center()
        top_layer.write("corrected.json")

    t_c = np.array(read_json('translation.json')['translation_vector']).astype(float)

    d0 = initial_displacement(atoms, distance)
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])
    w = maxz - minz
    atoms.cell[2, 2] += vacuum + w
    atoms.cell[2, 0:2] = 0.0
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

    calc = calc_setup(settings)

    def energy_fn(h):
        return get_energy(atoms, top_layer,
                          h, t_c, calc,
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

    final_atoms.write(outputname)

    curve = np.array(energy_curve)

    P = bilayer_stiffness(curve)

    results = {'heights': curve[:, 0],
               'energies': curve[:, 1],
               'optimal_height': hmin,
               'energy': opt_result.fun,
               'curvature': P[2],
               'FittingParams': P}

    return RelaxBilayerResult.fromdata(**results)


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
