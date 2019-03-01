import os

import numpy as np
from gpaw import GPAW
import gpaw.mpi as mpi
from ase.io import read
from ase.geometry import crystal_structure_from_cell

from ase.dft.kpoints import special_paths
from c2db.utils import get_special_2d_path, eigenvalues, gpw2eigs
creates = ['bs.gpw', 'eigs_spinorbit.npz']
dependencies = ['gs.py']


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


def get_parser():
    import argparse
    desc = 'Calculate electronic band structure'
    parser = argparse.ArgumentParser(description=desc)
    return parser


def main(args=None):
    parser = get_parser()
    args = vars(parser.parse_args(args))
    bandstructure(**args)


if __name__ == '__main__':
    main()
