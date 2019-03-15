import argparse

from pathlib import Path

import numpy as np

import ase.io.ulm as ulm
from ase.parallel import world
from ase.geometry import crystal_structure_from_cell
from ase.dft.kpoints import special_paths, bandpath
from ase.io import read
from ase.phonons import Phonons
from gpaw import GPAW


def phonons(N=2):
    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob('phonon.*.pckl'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    params = {}
    name = 'start.traj'

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}
    # Make sure to converge forces! Can be important
    if 'convergence' in params:
        params['convergence']['forces'] = 1e-6
    else:
        params['convergence'] = {'forces': 1e-6}

    slab = read(name)
    fd = open('phonons-{}.txt'.format(N), 'a')
    calc = GPAW(txt=fd, **params)

    # Set initial magnetic moments
    from rmr.utils import is_magnetic
    if is_magnetic():
        gsold = GPAW('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        slab.set_initial_magnetic_moments(magmoms_m)
        
    p = Phonons(slab, calc, supercell=(N, N, N))
    p.run()

    return p


def analyse(atoms, name='phonon', points=300, modes=False):
    state = Path().cwd().parts[-1]
    params = {}
    name = '../relax-{}.traj'.format(state)
    u = ulm.open(name)
    params.update(u[-1].calculator.parameters)
    u.close()
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}

    slab = read(name)
    calc = GPAW(txt='phonons.txt', **params)
    p = Phonons(slab, calc, supercell=(2, 2, 2))
    p.read(symmetrize=0, acoustic=False)
    cell = atoms.get_cell()
    cs = crystal_structure_from_cell(cell)
    kptpath = special_paths[cs]
    q_qc = bandpath(kptpath, cell, points)[0]

    out = p.band_structure(q_qc, modes=modes, born=False, verbose=False)
    if modes:
        omega_kl, u_kl = out
        return np.array(omega_kl), u_kl, q_qc
    else:
        omega_kl = out
        return np.array(omega_kl), np.array(omega_kl), q_qc


def main(args):
    phonons(**args)


short_description = 'Calculate phonons'
dependencies = ['rmr.gs']
parser = argparse.ArgumentParser(description=short_description)
parser.add_argument('-N', help='Super-cell size', type=int, default=2)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
