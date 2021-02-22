from ase.io import read
from ase.units import Bohr
from asr.core import command, option, ASRResult
from gpaw import GPAW, restart, setup_paths
from gpaw.utilities.ps2ae import PS2AE

from typing import List, Dict, Any, Optional
from math import sqrt, pi
import numpy as np

setup_paths[:0] = ['..']

@command('asr.nonradiative')
@option('--folder', help='Folder of the displaced geometry', type=str)
@option('--npoints', help='Number of displacement points.', type=int)
@option('--percent', type=float,
        help='Percent of displacement (0.2 is recommended)') 
@option('--initial',
        help='Band indices of the initial state/es')
@option('--final', type=int,
        help='Band index of the final defect state')
@option('--spin', help='Spin channel index.', type=int)
def calculate(folder: str, initial: List[int], final: int, spin: int = 0, 
              npoints: int = 5, percent: float = 0.2): 
    """The electron-phonon matrix element is calculated between
       the initial and final state along the 1d-dimensional mode."""

    atoms, calc_0 = restart('0.0-wf.gpw', txt=None)
    atoms_Q = read(folder + '/gs.gpw')

    displ_n = np.linspace(-percent, percent, npoints, endpoint=True)
    m_a = atoms.get_masses()
    pos_ai = atoms.positions.copy()

    # define the 1D coordinate
    delta_R = atoms_Q.positions - atoms.positions
    delta_Q = sqrt(((delta_R**2).sum(axis=-1) * m_a).sum())

    # do a fixed density calculation first
    #calc_0 = calc_0.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})

    # set up calculator from the Q=0 geometry
    #params = calc_0.todict()
    #calc_q = GPAW(**params)
    #calc_q.set(txt='1d-elph.txt')
    #calc_q.set(symmetry='off')

    # quantities saved along the displacement
    Q_n = []
    #eigenvalues_ni = np.array((npoints, len(initial)))
    #overlap_ni = np.array((npoints, len(initial)))

    wfs_0 = PS2AE(calc_0, grid_spacing=0.05)
    wf_0 = wfs_0.get_wave_function(n=final, s=spin)
    displ_n = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    for n, displ in enumerate(displ_n):
        print(displ, f'{displ}-wf.gpw')
        #Q2 = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        #Q_n.append(sqrt(Q2) * np.sign(displ))
        #atoms.positions += displ * delta_R

        calc_q = GPAW(f'{displ}-wf.gpw')
        wfs_q = PS2AE(calc_q, grid_spacing=0.05)
        wf_q = wfs_q.get_wave_function(n=final, s=spin)

        phase = wfs_0.gd.integrate(wf_0 * wf_q) * Bohr**3
        sign = np.sign(phase)
        #atoms.set_calculator(calc_q)

        for i in initial:
            wf_q = wfs_q.get_wave_function(n=i, s=spin)
            overlap = wfs_0.gd.integrate(wf_0 * wf_q) * Bohr**3
            ei = calc_q.get_eigenvalues(kpt=0, spin=spin)[i]
            print(i, final, overlap, sign * overlap, ei)
            #overlap, eigenvalue = wfs_overlap(i, f, calc_0, calc_q)
            #overlap_ni[n,i] = overlap
            #eigenvalues_ni[n,i] = eigenvalue


def main():
    """The nonradiative rate as function of the temperature is 
       calculated within the one-dimensional approximation. It
       follows the method in of Phys.Rev. B 90, 075202 (2014).
       and the implemetation described in the nonrad code."""


def webpanel(result, row, key_descriptions):
    from asr.browser import fig, table

    if 'something' not in row.data:
        return None, []

    table1 = table(row,
                   'Property',
                   ['something'],
                   kd=key_descriptions)
    panel = ('Title',
             [[fig('something.png'), table1]])
    things = [(create_plot, ['something.png'])]
    return panel, things


if __name__ == '__main__':
    main()
