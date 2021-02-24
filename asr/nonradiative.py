from asr.core import read_json, command, option, ASRResult, prepare_result

from ase.io import read
from ase.units import Bohr
import ase.units as units

from gpaw import GPAW, restart, setup_paths
from gpaw.utilities.ps2ae import PS2AE

from scipy.special import eval_hermite as herm
from scipy.interpolate import interp1d
from scipy import constants as const

from math import sqrt, pi, factorial

from typing import List, Dict, Any, Tuple, Optional
from math import sqrt, pi
import numpy as np

setup_paths[:0] = ['..']


@prepare_result
class ElphResults(ASRResult):
    """Container for the electron-phonon results."""
    Q_n: np.ndarray
    Wif_i: np.ndarray
    eigenvalues_ni: np.ndarray
    overlap_ni: np.ndarray

    key_descriptions = dict(
        Q_n='Displacement coordinate along 1d phonon mode [A]',
        Wif_i='Electron-Phonon matrix elements for the different' 
               'bands [eV / amu^1/2 / A]',
        eigenvalues_ni='Eigenvalues for different displacements. [eV]',
        overlap_ni='Overlaps for different displacements.')


@prepare_result
class RateResults(ASRResult):
    """Container for the nonradiative results."""
    T_n: np.ndarray
    C_n: np.ndarray

    key_descriptions = dict(
        T_n='Range of temperatures [K].',
        C_n='Nonradiative rate as function of the temperature [cm^3 / s].')


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
              npoints: int = 5, percent: float = 0.2) -> ElphResults: 
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
    Q_n = np.empty((npoints))
    eigenvalues_ni = np.empty((npoints, len(initial)))
    overlap_ni = np.empty((npoints, len(initial)))
    displ_n = np.array([-0.2,-0.1,0.0,0.1,0.2])
    Wif_i = np.empty((npoints))

    calc_0 = GPAW('0.0-wf.gpw')

    wfs_0 = PS2AE(calc_0, grid_spacing=0.05)
    ef = calc_0.get_eigenvalues(kpt=0, spin=spin)[final]

    for n, displ in enumerate(displ_n):
        Q2 = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        Q_n[n] = sqrt(Q2) * np.sign(displ)
        atoms.positions += displ * delta_R

        calc_q = GPAW(f'{displ}-wf.gpw')
        wfs_q = PS2AE(calc_q, grid_spacing=0.05)
        wf_q = wfs_q.get_wave_function(n=final, s=spin)
        wf = wfs_0.get_wave_function(n=final, s=spin)
        # sign for the diagonal overlap
        sign = np.sign(wfs_0.gd.integrate(wf * wf_q) * Bohr**3)

        for i, band in enumerate(initial):
            wf_0 = wfs_0.get_wave_function(n=band, s=spin)
            overlap = wfs_0.gd.integrate(wf_0 * wf_q) * Bohr**3

            ei = calc_q.get_eigenvalues(kpt=0, spin=spin)[band]
            delta_if = ef - ei
            print(band, final, sign * overlap, ei)

            eigenvalues_ni[n,i] = ei
            overlap_ni[n,i] = sign * overlap

    for i, overlap_i in enumerate(overlap_ni.T):
        ei = calc_0.get_eigenvalues(kpt=0, spin=spin)[initial[i]]
        delta = ef - ei
        w = np.polyfit(displ_n, overlap_i, 1)
        Wif_i[i] = (ef - ei) * w[0]


    return ElphResults.fromdata(
        Q_n=Q_n,
        Wif_i=Wif_i,
        eigenvalues_ni=eigenvalues_ni,
        overlap_ni=overlap_ni
        )


@command('asr.nonradiative')
@option('--temperatures', nargs=2, type=float, 
        help='Range of temperatures for the rate [K]') 
@option('--sigma', type=float, 
        help='Smearing for the delta function [eV].') 
@option('--g', type=int, 
        help='Configurational degeneracy') 
@option('--wif', type=float, 
        help='Electron-Phonon coupling to the 1d mode [eV / amu^1/2 / A]') 
@option('--frequencies', nargs=2, type=float, 
        help='Frequencies of the 1d mode in the initial and final state [eV]') 
@option('--delta_q', type=float, 
        help='Change in the geometries [amu^1/2 A]') 
@option('--delta_e', type=float,
        help='Charge transition level [eV]') 
def main(delta_e: float, delta_q: float = None,
         frequencies: Tuple[float, float] = None, wif: float = None,
         g: int = 1, sigma: float = None,
         temperatures: Tuple[float, float] = [25,800]):
    """The nonradiative rate as function of the temperature is 
       calculated within the one-dimensional approximation. It
       follows the method in Phys.Rev. B 90, 075202 (2014) and
       the implemetation described in arXiv:2011.07433."""

    print('Start')

    elphparams = read_json('results-asr.nonradiative@calculate.json').metadata.params

    #atoms_i = read(elphparams['folder'] + 'structure.json')
    atoms_f = read('structure.json')

    Omegai, Omegaf = frequencies

    volume = atoms_f.get_volume()

    T_n = np.linspace(temperatures[0], temperatures[-1], 1000)

    C_n = get_rate(delta_e, delta_q, Omegai, Omegaf, g, wif, volume, T_n)

    return RateResults.fromdata(
        T_n=T_n,
        C_n=C_n)

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


def get_rate(delta_E, delta_Q, Omegai, Omegaf, g, Wif, volume, sigma, T_n):

    s2 = units._hbar / 1e-20 / units._amu
    s3 = units._e / units._hbar

    Ni = 17
    Nf = 50

    q_i = np.linspace(-30,30,1000)
    C_n = 0

    for m in range(Ni):
        Z = 1. / (1 - np.exp(-Omegai / (units.kB * T_n)))
        weight = np.exp(-m * Omegai / (units.kB * T_n) ) / Z
        for n in range(Nf):
            if m == 0:
                overlplus = overlap(q_i, 1, Omegai, n, Omegaf, delta_Q)
                overlequal = overlap(q_i, 0, Omegai, n, Omegaf, delta_Q)

                Overlap = sqrt(s2 / 2 / Omegai) * overlplus
                Overlap += sqrt(s3) * delta_Q * overlequal
                #print('my: ', n, Overlap)
            else:
                overlplus = overlap(q_i, m+1, Omegai, n, Omegaf, delta_Q)
                overlequal = overlap(q_i, m, Omegai, n, Omegaf, delta_Q)
                overlminus = overlap(q_i, m-1, Omegai, n, Omegaf, delta_Q)

                Overlap = sqrt(s2 / 2 / Omegai * (m+1)) * overlplus
                Overlap += sqrt(s2 / 2 / Omegai * m) * overlminus
                Overlap += sqrt(s3) * delta_Q * overlequal
            arg_delta = delta_E + m * Omegai - n * Omegaf
            C_n += weight * Overlap**2 * gauss_delta(arg_delta, sigma)

    return 2 * np.pi * g * Wif**2 * volume * (1e-8)**3 * C_n


def gauss_delta(x, sigma):
    return 1 / (sqrt(2 * pi * sigma**2)) * np.exp(-x**2 / (2 * sigma**2))


def overlap(q_i, n1, Omega1, n2, Omega2, delta_Q):
    wf1_i = vibronic_wf(q_i - delta_Q, n1, Omega1)
    wf2_i = vibronic_wf(q_i, n2, Omega2)
    overlap = np.trapz( wf1_i * wf2_i, x = q_i)
    return overlap


def vibronic_wf(x, n, w):
    s = units._e * units._amu * 1e-20 / units._hbar**2
    normalization = 1 / sqrt(2**n * factorial(n)) * (s * w / pi)**0.25
    wf = normalization * np.exp(-s * w * x**2 / 2) * herm(n, sqrt(s * w) * x)
    return wf


if __name__ == '__main__':
    main()
