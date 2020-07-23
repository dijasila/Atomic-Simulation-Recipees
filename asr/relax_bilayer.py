from asr.core import command, option, AtomsFile, DictStr
import numpy as np
from ase import Atoms


def get_energy(base, top, h, t_c, settings, callback):
    from ase.calculators.dftd3 import DFTD3
    from gpaw import GPAW, PW
    from asr.stack_bilayer import translation

    calc = GPAW(mode=PW(800),
                xc=settings['xc'],
                kpts=settings['kpts'],
                symmetry={'symmorphic': False},
                occupations={'name': 'fermi-dirac',
                             'width': 0.05},
                charge=0,
                txt='relax_bilayer.txt')
    if settings['d3']:
        calc = DFTD3(dft=calc, cutoff=60)

    tx, ty = t_c[0], t_c[1]
    atoms = translation(tx, ty, h[0], top, base)
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()

    callback(h, e)

    return e


def initial_displacement(atoms):
    maxz = np.max(atoms.positions[:, 2])
    minz = np.min(atoms.positions[:, 2])

    d0 = 3

    return d0 + (maxz - minz)


@command('asr.relax_bilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-s', '--settings', help='Relaxation settings',
        type=DictStr())
@option('-n', '--name', help='Name of final structure file',
        default='structure.json')
def main(atoms: Atoms,
         settings: dict = {'d3': True,
                           'xc': 'PBE',
                           'mode': 'interlayer',
                           'PWE': 800,
                           'kpts': {'density': 6.0, 'gamma': True}},
         name='structure.json'):
    from asr.core import read_json
    from ase.io import read
    from gpaw import mpi
    import scipy.optimize as sciop
    from asr.stack_bilayer import translation

    top_layer = read('toplayer.json')

    t_c = read_json('translation.json')['translation_vector'].astype(float)

    energy_curve = []

    def callback_fn(h, energy):
        energy_curve.append((h, energy))
        if mpi.rank == 0:
            np.save('energy_curve.npy', np.array(energy_curve))

    def energy_fn(h):
        return get_energy(atoms, top_layer,
                          h, t_c, settings, callback_fn)

    d0 = initial_displacement(atoms)

    opt_result = sciop.minimize(energy_fn, x0=d0, method="Nelder-Mead")
    if mpi.rank == 0:
        np.save('energy_curve.npy', np.array(energy_curve))

    if not opt_result.success:
        raise ValueError('Relaxation failed')

    hmin = opt_result.x

    final_atoms = translation(t_c[0], t_c[1], hmin,
                              top_layer, atoms)

    final_atoms.write(name)

    curve = np.array(energy_curve)

    return {'heights': curve[:, 0],
            'energies': curve[:, 1]}
