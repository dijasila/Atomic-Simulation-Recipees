import argparse

import pickle
from pathlib import Path
from itertools import product

import numpy as np

import ase.io.ulm as ulm
import ase.units as units
from ase.parallel import world
from ase.geometry import crystal_structure_from_cell
from ase.dft.kpoints import special_paths, bandpath
from ase.io import read
from ase.phonons import Phonons
from gpaw import GPAW


def phonons(N=2):
    state = Path().cwd().parts[-1]

    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob('phonon.*.pckl'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    params = {}
    name = 'start.traj'.format(state)

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}
    # Make sure to converge forces! Can be important
    if 'convergence' in params:
        params['convergence']['forces'] = 1e-6
    else:
        params['convergence'] = {'forces': 1e-6}

    oldcalc = GPAW('gs.gpw', txt=None)
    N_c = (oldcalc.wfs.gd.N_c // 4) * 4 * N
    params['gpts'] = N_c
    slab = read(name)

    fd = open('phonons-{}.txt'.format(N), 'a')
    calc = GPAW(txt=fd, **params)

    # Set initial magnetic moments
    if state[:2] == 'fm' or state[:3] == 'afm':
        gsold = GPAW('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        slab.set_initial_magnetic_moments(magmoms_m)
        
    p = Phonons(slab, calc, supercell=(N, N, N))
    p.run()


def analyse2(atoms, name='phonon', points=300, modes=False):
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

            
def analyse(atoms, name='phonon', points=100):
    D = 3
    delta = 0.01
    N = len(atoms)
    C = np.empty((2, 2, 2, N, D, 2, 2, 2, N, D))
    cell = atoms.get_cell()
    cs = crystal_structure_from_cell(cell)
    kptpath = special_paths[cs]
    q_qc = bandpath(kptpath, cell, points)[0]

    for a in range(N):
        for i, v in enumerate('xyz'[:D]):
            forces = []
            for sign in '-+':
                filename = '{}.{}{}{}.pckl'.format(name, a, v, sign)
                with open(filename, 'rb') as fd:
                    f = pickle.load(fd)[:, :D]
                    f[a] -= f.sum(axis=0)
                    f.shape = (2, 2, 2, N, D)
                    forces.append(f)
            C[0, 0, 0, a, i] = (forces[0] - forces[1]) / (2 * delta)

    for i, j, k in product([0, 1], repeat=3):
        if i == j == k == 0:
            continue

        # This will only work for 2x repeat directions
        C[i, j, k] = C[0, 0, 0, :, :,
                       ::((i == 0) - (i == 1)),
                       ::((j == 0) - (j == 1)),
                       ::((k == 0) - (k == 1))]
            
    C.shape = (8 * D * N, 8 * D * N)
    C += C.T.copy()
    C *= 0.5

    # Mingo correction.
    #
    # See:
    #
    #    Phonon transmission through defects in carbon nanotubes
    #    from first principles
    #
    #    N. Mingo, D. A. Stewart, D. A. Broido, and D. Srivastava
    #    Phys. Rev. B 77, 033418 – Published 30 January 2008
    #    http://dx.doi.org/10.1103/PhysRevB.77.033418

    R_in = np.zeros((8 * D * N, D))
    for n in range(D):
        R_in[n::D, n] = 1.0
    a_in = -np.dot(C, R_in)
    B_inin = np.zeros((8 * D * N, D, 8 * D * N, D))
    for i in range(8 * D * N):
        B_inin[i, :, i] = np.dot(R_in.T, C[i, :, np.newaxis]**2 * R_in) / 4
        for j in range(4 * D * N):
            B_inin[i, :, j] += np.outer(R_in[i], R_in[j]).T * C[i, j]**2 / 4
    L_in = np.dot(np.linalg.pinv(B_inin.reshape((8 * D**2 * N, 8 * D**2 * N))),
                  a_in.reshape((8 * D**2 * N,))).reshape((8 * D * N, D))
    D_ii = C**2 * (np.dot(L_in, R_in.T) + np.dot(L_in, R_in.T).T) / 4
    C += D_ii

    # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
    s = units._hbar * 1e10 / (units._e * units._amu)**0.5
    M = np.repeat(atoms.get_masses(), D)
    invM = np.outer(M, M)**-0.5
    eigs = []
    freqs = []
    C.shape = (2, 2, 2, N * D, 2, 2, 2, N * D)
    for q_c in q_qc:
        K = np.zeros((N * D, N * D), dtype=complex)
        for c1 in range(2):
            for c2 in range(2):
                for c3 in range(2):
                    K += (np.exp(-2j * np.pi * np.dot(q_c, (c1, c2, c3))) *
                          C[0, 0, 0, :, c1, c2, c3])
        e = np.linalg.eigvalsh(K)
        f2 = np.linalg.eigvalsh(invM * K)
        f = abs(f2)**0.5 * np.sign(f2) * s
        eigs.append(e)
        freqs.append(f)

    return np.array(eigs), np.array(freqs), q_qc


def main(args):
    phonons(**args)


short_description = 'Calculate phonons'
dependencies = ['rmr.gs']
parser = argparse.ArgumentParser(description=short_description)
parser.add_argument('-N', help='Super-cell size', type=int, default=2)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
