import argparse

import os
import os.path as op

import numpy as np
from gpaw import GPAW
import gpaw.mpi as mpi
from ase.io import read
from ase.geometry import crystal_structure_from_cell

from ase.dft.kpoints import special_paths
from c2db.utils import get_special_2d_path, eigenvalues, gpw2eigs, spin_axis

creates = ['bs.gpw', 'eigs_spinorbit.npz']
dependencies = ['asr.gs']


def gs_done():
    return os.path.isfile('gs.gpw')


def bs_done():
    return os.path.isfile('bs.gpw')


def bandstructure(kptpath=None, npoints=400, emptybands=20):
    """Calculate the bandstructure based on a relaxed structure in gs.gpw."""

    if os.path.isfile('eigs_spinorbit.npz'):
        return
    if not gs_done():
        return
    if not bs_done():
        if kptpath is None:
            atoms = read('gs.gpw')
            cell = atoms.cell
            ND = np.sum(atoms.pbc)
            if ND == 3:
                cs = crystal_structure_from_cell(cell)
                kptpath = special_paths[cs]
            elif ND == 2:
                kptpath = get_special_2d_path(cell)
            else:
                raise NotImplementedError
            
        convbands = emptybands // 2
        parms = {'basis': 'dzp',
                 'nbands': -emptybands,
                 'txt': 'bs.txt',
                 'fixdensity': True,
                 'kpts': {'path': kptpath, 'npoints': npoints},
                 'convergence': {'bands': -convbands},
                 'symmetry': 'off'}

        calc = GPAW('gs.gpw',
                    **parms)

        calc.get_potential_energy()
        calc.write('bs.gpw')

    calc = GPAW('bs.gpw', txt=None)
    path = calc.get_bz_k_points()

    # stuff below could be moved to the collect script.
    e_nosoc_skn = eigenvalues(calc)
    e_km, _, s_kvm = gpw2eigs('bs.gpw', soc=True, return_spin=True,
                              optimal_spin_direction=True)
    if mpi.world.rank == 0:
        with open('eigs_spinorbit.npz', 'wb') as f:
            np.savez(f, e_mk=e_km.T, s_mvk=s_kvm.transpose(2, 1, 0),
                     e_nosoc_skn=e_nosoc_skn, path=path)


def is_symmetry_protected(kpt, op_scc):
    mirror_count = 0
    for symm in op_scc:
        # Inversion symmetry forces spin degeneracy and 180 degree rotation
        # forces the spins to lie in plane
        if (np.allclose(symm, -1 * np.eye(3)) or
                np.allclose(symm, np.array([-1, -1, 1] * np.eye(3)))):
            return True
        vals, vecs = np.linalg.eigh(symm)
        # A mirror plane
        if np.allclose(np.abs(vals), 1) and np.allclose(np.prod(vals), -1):
            # Mapping k -> k, modulo a lattice vector
            if np.allclose(kpt % 1, (np.dot(symm, kpt)) % 1):
                mirror_count += 1
    # If we have two or more mirror planes, then we must have a spin-degenerate
    # subspace
    if mirror_count >= 2:
        return True
    return False


def collect_data(kvp, data, atoms, verbose):
    """Band structure PBE and GW +- SOC."""
    if not op.isfile('eigs_spinorbit.npz'):
        return
    print('Collecting PBE bands-structure data')
    evac = kvp.get('evac')
    with open('eigs_spinorbit.npz', 'rb') as fd:
        soc = dict(np.load(fd))
    if 'e_nosoc_skn' in soc and 'path' in soc:
        eps_skn = soc['e_nosoc_skn']
        path = soc['path']
        # atoms = read('bs.gpw')
    else:
        nosoc = GPAW('bs.gpw', txt=None)
        # atoms = nosoc.get_atoms()
        eps_skn = eigenvalues(nosoc)
        path = nosoc.get_bz_k_points()
    npoints = len(path)
    s_mvk = soc.get('s_mvk', soc.get('s_mk'))
    if s_mvk.ndim == 3:
        sz_mk = s_mvk[:, spin_axis(), :]  # take x, y or z component
        if sz_mk.shape.index(npoints) == 0:
            sz_mk = sz_mk.transpose()
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, 'sz_mk has wrong dims'

    efermi = np.load('gap_soc.npz')['efermi']
    efermi_nosoc = np.load('gap.npz')['efermi']

    pbe = {
        'path': path,
        'eps_skn': eps_skn - evac,
        'efermi_nosoc': efermi_nosoc - evac,
        'efermi': efermi - evac,
        'eps_so_mk': soc['e_mk'] - evac,
        'sz_mk': sz_mk}
    try:
        op_scc = data['op_scc']
    except KeyError:
        from gpaw.symmetry import atoms2symmetry
        op_scc = atoms2symmetry(atoms).op_scc
    magstate = kvp['magstate']
    for idx, kpt in enumerate(path):
        if (magstate == 'NM' and is_symmetry_protected(kpt, op_scc) or
                magstate == 'AFM'):
            pbe['sz_mk'][:, idx] = 0.0

    data['bs_pbe'] = pbe


def main(args):
    bandstructure(**args)


short_description = 'Calculate electronic band structure'
parser = argparse.ArgumentParser(description=short_description)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
