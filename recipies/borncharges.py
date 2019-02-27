# Creates: borncharges-0.01.json, borncharges-0.005.json

import json
from os.path import exists, splitext, isfile
from os import remove, chdir, makedirs
from glob import glob

import numpy as np
from gpaw import GPAW
from gpaw.mpi import world
from c2db.berryphase import get_polarization_phase

from ase.parallel import paropen
from ase.units import Bohr
from ase.io import jsonio


def get_wavefunctions(atoms, name, params, density=6.0):
    params['kpts'] = {'density': density,
                      'gamma': True,
                      'even': True}
    params['symmetry'] = {'point_group': True,
                          'do_not_symmetrize_the_density': False,
                          'time_reversal': True}
    params['convergence']['eigenstates'] = 1e-11
    tmp = splitext(name)[0]
    atoms.calc = GPAW(txt=tmp + '.txt', **params)
    atoms.get_potential_energy()
    atoms.calc.write(name, 'all')
    return atoms.calc


def borncharges(displacement=0.01, kpointdensity=6.0, folder=None):

    if folder is None:
        folder = 'data-borncharges'
    if world.rank == 0:
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    world.barrier()
    
    chdir(folder)
    calc = GPAW('../gs.gpw', txt=None)
    params = calc.parameters
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    pos_av = atoms.get_positions().copy()
    avg_v = np.sum(pos_av, axis=0) / len(pos_av)
    pos_av -= avg_v
    atoms.set_positions(pos_av)
    Z_avv = []
    P_asvv = []
    
    if world.rank == 0:
        print('Atomnum Atom Direction Displacement')
    for a in range(len(atoms)):
        phase_scv = np.zeros((2, 3, 3), float)
        for v in range(3):
            for s, sign in enumerate([-1, 1]):
                if world.rank == 0:
                    print(sym_a[a], a, v, s)
                # Update atomic positions
                atoms.positions = pos_av
                atoms.positions[a, v] = pos_av[a, v] + sign * displacement
                prefix = 'born-{}-{}{}{}'.format(displacement, a,
                                                 'xyz'[v],
                                                 ' +-'[sign])
                name = prefix + '.gpw'
                berryname = prefix + '-berryphases.json'
                if not exists(name) and not exists(berryname):
                    calc = get_wavefunctions(atoms, name, params,
                                             density=kpointdensity)

                try:
                    phase_c = get_polarization_phase(name)
                except ValueError:
                    calc = get_wavefunctions(atoms, name, params,
                                             density=kpointdensity)
                    phase_c = get_polarization_phase(name)

                phase_scv[s, :, v] = phase_c

                if exists(berryname):  # Calculation done?
                    if world.rank == 0:
                        # Remove gpw file
                        if isfile(name):
                            remove(name)

        dphase_cv = (phase_scv[1] - phase_scv[0])
        mod_cv = np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dphase_cv -= mod_cv
        phase_scv[1] -= mod_cv
        dP_vv = (-np.dot(dphase_cv.T, cell_cv).T /
                 (2 * np.pi * vol))

        P_svv = (-np.dot(cell_cv.T, phase_scv).transpose(1, 0, 2) /
                 (2 * np.pi * vol))
        Z_vv = dP_vv * vol / (2 * displacement / Bohr)
        P_asvv.append(P_svv)
        Z_avv.append(Z_vv)

    data = {'Z_avv': Z_avv, 'sym_a': sym_a,
            'P_asvv': P_asvv}

    filename = 'borncharges-{}.json'.format(displacement)

    with paropen(filename, 'w') as fd:
        json.dump(jsonio.encode(data), fd)

    world.barrier()
    if world.rank == 0:
        files = glob('born-*.gpw')
        for f in files:
            if isfile(f):
                remove(f)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Born charges')
    parser.set_defaults(func=main)

    # Set variables
    help = 'Atomic displacement when moving atoms in Å'
    parser.add_argument('-d', '--displacement', help=help, default=0.01)
    help = 'Set kpoint density for calculation of berryphases'
    parser.add_argument('-k', '--kpointdensity', help=help, default=6.0)
    help = 'Folder where data is put'
    parser.add_argument('-f', '--folder', help=help,
                        default='data-borncharges')
    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop('func', None)
    borncharges(**kwargs)


if __name__ == '__main__':
    main()
