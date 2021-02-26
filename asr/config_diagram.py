from typing import Tuple
import click
from ase.parallel import parprint
from asr.core import command, option, read_json, ASRResult, prepare_result
import ase.units as units
from math import sqrt, pi, factorial
import numpy as np


def get_wfs_overlap(i, f, calc_0, calc_q):
    """ Calculate the overlap between that the state i
        and the state f of the displaced geometry"""

    from gpaw.utilities.ps2ae import PS2AE
    from ase.units import Bohr

    wfs_0 = PS2AE(calc_0, h=0.05)
    wfs_q = PS2AE(calc_q, h=0.05)

    wf_0 = wfs_0.get_wave_function(i)
    wf_q = wfs_q.get_wave_function(f)

    overlap = wfs_q.gd.integrate(wf_0 * wf_q) * Bohr**3

    ei = calc_q.get_eigenvalues(kpt=0, spin=0)[i]
    ef = calc_q.get_eigenvalues(kpt=0, spin=0)[f]

    eigenvalues = [ei, ef]

    return overlap, eigenvalues


@prepare_result
class DisplacementResults(ASRResult):
    """Container for results related to the displaced geometries."""
    delta_Q: float
    Q_n: np.ndarray
    energies_n: np.ndarray
    ZPL: float
    overlap: np.ndarray
    eigenvalues: np.ndarray

    key_descriptions = dict(
        delta_Q='1D displacement coordinate along main phonon mode [Å].',
        Q_n='Displacements array along 1D coordinate [Å].',
        energies_n='Energies along displaced coordinates [eV].',
        ZPL='Zero phonon line energy [eV].',
        overlap='Overlap between state i and state f of the displaced geometry.',
        eigenvalues='Eigenvalues for state i and f of the displaced geometry [eV].')


@command("asr.config_diagram",
         returns=DisplacementResults)
@option('--folder', help='Folder of the displaced geometry', type=str)
@option('--npoints', help='How many displacement points.', type=int)
@option('--wfs', nargs=2, type=click.Tuple([int, int]),
        help='Calculate the overlap of wfs between states i and f')
def calculate(folder: str, npoints: int = 5,
              wfs: Tuple[int, int] = None) -> DisplacementResults:
    """Interpolate the geometry of the structure in the current folder with the
       displaced geometry in the 'folder' given as input of the recipe.
       The number of displacements between the two geometries is set with the
       'npoints' input, and the energy, the modulus of the displacement and
       the overlap between the wavefunctions is saved (if wfs is set)."""

    from gpaw import GPAW, restart

    atoms, calc_0 = restart('gs.gpw', txt=None)
    atoms_Q, calc_Q = restart(folder + '/gs.gpw', txt=None)

    # calculate the Zero Phonon Line
    zpl = abs(atoms.get_potential_energy() - atoms_Q.get_potential_energy())

    # percent of displacement from -100% to 100% with npoints
    displ_n = np.linspace(-1.0, 1.0, npoints, endpoint=True)
    m_a = atoms.get_masses()
    pos_ai = atoms.positions.copy()

    # define the 1D coordinate
    delta_R = atoms_Q.positions - atoms.positions
    delta_Q = sqrt(((delta_R**2).sum(axis=-1) * m_a).sum())

    # check if there is difference in the two geometries
    assert delta_Q >= 0.005, 'No displacement between the two geometries!'
    parprint('delta_Q', delta_Q, zpl)

    # if overlap is calculated do a fixed density calculation first
    if wfs:
        calc_0 = calc_0.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})

    # set up calculator from the Q=0 geometry
    params = calc_0.todict()
    calc_q = GPAW(**params)
    calc_q.set(txt='cc-diagram.txt')
    calc_q.set(symmetry='off')

    # quantities saved along the displacement
    Q_n = []
    energies_n = []
    overlap_n = []
    eigenvalues_n = []

    for displ in displ_n:
        Q2 = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        Q_n.append(sqrt(Q2) * np.sign(displ))
        atoms.positions += displ * delta_R
        atoms.set_calculator(calc_q)
        # energy = atoms.get_potential_energy()
        # parprint(displ, energy)

        atoms.positions = pos_ai
        energy = 0.06155**2 / 2 * 15.4669**2 * Q2
        energies_n.append(energy)

        # if the indices of the wfs are set the overlap is calculated
        if wfs:
            i, f = wfs
            overlap, eigenvalues = get_wfs_overlap(i, f, calc_0, calc_q)
            parprint(overlap, eigenvalues)
        else:
            overlap = None
            eigenvalues = None
        overlap_n.append(overlap)
        eigenvalues_n.append(eigenvalues)

    return DisplacementResults.fromdata(
        delta_Q=delta_Q,
        Q_n=Q_n,
        energies_n=energies_n,
        ZPL=zpl,
        overlap=overlap_n,
        eigenvalues=eigenvalues_n)


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Configuration coordinate diagram',
             'columns': [[fig('cc_diagram.png')],
                         [fig('luminescence.png')]],
             'plot_descriptions': [{'function': plot_cc_diagram,
                                    'filenames': ['cc_diagram.png']},
                                   {'function': plot_luminescence,
                                    'filenames': ['luminescence.png']}],
             'sort': 12}

    return [panel]


@prepare_result
class ParabolaResults(ASRResult):
    """Container for frequencies, energies, and Huang-Rhys factor of
    excited state or ground state."""
    energies_n: np.ndarray
    omega: float
    S: float

    key_descriptions = dict(
        energies_n='Energies along displacement path [eV].',
        omega='Effective frequency [eV].',
        S='Huang-Rhys factor.')


@prepare_result
class Result(ASRResult):
    """Container for configuration diagram results."""
    Q_n: np.ndarray
    ZPL: float
    delta_Q: float
    ground: ParabolaResults
    excited: ParabolaResults

    key_descriptions = dict(
        Q_n='Displacements array along 1D coordinate [Å].',
        ZPL='Zero phonon line energy [eV].',
        delta_Q='1D displacement coordinate along main phonon mode [Å].',
        ground='Ground state ParabolaResults.',
        excited='Excited state ParabolaResults.')


def return_parabola_results(energies_n, omega, S):
    return ParabolaResults.fromdata(
        energies_n=energies_n,
        omega=omega,
        S=S)


@command("asr.config_diagram",
         dependencies=["asr.config_diagram@calculate"],
         returns=Result)
@option('--folder1', help='Folder of the first parabola', type=str)
@option('--folder2', help='Folder of the first parabola', type=str)
def main(folder1: str = '.', folder2: str = 'excited') -> Result:
    """Estrapolate the frequencies of the ground and
       excited one-dimensional mode and their relative
       Huang-Rhys factors"""

    result_file = 'results-asr.config_diagram@calculate.json'

    data_g = read_json(folder1 + '/' + result_file)
    data_e = read_json(folder2 + '/' + result_file)
    delta_Q = data_g['delta_Q']
    energies_gn = data_g['energies_n']
    energies_en = data_e['energies_n']
    Q_n = data_g['Q_n']
    zpl = data_g['ZPL']

    # Rescale ground energies by the minimum value
    energies_gn = np.array(energies_gn)
    energies_gn -= np.min(energies_gn)
    # Rescale excited energies by the minimum value
    energies_en = np.array(energies_en)
    energies_en -= np.min(energies_en)

    # Quadratic fit of the parabola
    zg = np.polyfit(Q_n, energies_gn, 2)
    ze = np.polyfit(Q_n, energies_en, 2)
    # Conversion factor
    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    # Estrapolation of the effective frequencies
    omega_g = sqrt(2 * zg[0] / s**2)
    omega_e = sqrt(2 * ze[0] / s**2)

    # Estrapolation of the Huang-Rhys factors
    S_g = s**2 * delta_Q**2 * omega_g / 2
    S_e = s**2 * delta_Q**2 * omega_e / 2

    # return ParabolaResults for gs and excited state, respectively
    parabola_ground = return_parabola_results(energies_gn, omega_g, S_g)
    parabola_excited = return_parabola_results(energies_en, omega_e, S_e)

    return Result.fromdata(
        Q_n=Q_n,
        ZPL=zpl,
        delta_Q=delta_Q,
        ground=parabola_ground,
        excited=parabola_excited)


def plot_cc_diagram(row, fname):
    from matplotlib import pyplot as plt

    data = row.data.get('results-asr.config_diagram.json')
    data_g = data['ground']
    data_e = data['excited']

    ene_g = data_g['energies_n']
    ene_e = data_e['energies_n']

    Q_n = np.array(data['Q_n'])
    ZPL = data['ZPL']
    delta_Q = data['delta_Q']
    q = np.linspace(Q_n[0] - delta_Q * 0.2, Q_n[-1] + delta_Q * 0.2, 100)

    omega_g = data_g['omega']
    omega_e = data_e['omega']

    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()

    # Ground state parabola
    ax.plot(q, 1 / 2 * omega_g**2 * s**2 * q**2, '-C0')
    ax.plot(Q_n, ene_g, 'wo', ms=7, markeredgecolor='C0', markeredgewidth=0.9)
    # Excited state parabola
    ax.plot(q + delta_Q, 1 / 2 * omega_e**2 * s**2 * q**2 + ZPL, '-C1')
    ax.plot(Q_n + delta_Q, ene_e + ZPL, 'wo', ms=7,
            markeredgecolor='C1', markeredgewidth=0.9)

    ax.set_xlabel(r'Q$\;(amu^{1/2}\AA)$', size=14)
    ax.set_ylabel('Energy (eV)', size=14)
    ax.set_xlim(-1.3 * delta_Q, 2 * delta_Q * 1.15)
    ax.set_ylim(-1 / 5 * ZPL, 1.1 * max(ene_e) + 6 / 5 * ZPL)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_luminescence(row, fname):
    from matplotlib import pyplot as plt

    def gauss_delta(x, sigma):
        return 1 / (sigma * sqrt(2 * pi)) * np.exp(-x**2 / (2 * sigma**2))

    def overlap(n, S):
        return np.exp(-S) * S**n / factorial(n)

    data = row.data.get('results-asr.config_diagram.json')

    omega_g = data['ground']['omega']
    omega_e = data['excited']['omega']
    ZPL = data['ZPL']
    S_g = data['ground']['S']

    w_i = np.linspace(0, ZPL * 1.3, 1000)

    nmax = 20
    sigma = 0.02
    L_i = 0

    for n in range(1, nmax):
        arg_delta = ZPL + omega_e - n * omega_g - w_i
        L_i += overlap(n, S_g) * gauss_delta(arg_delta, sigma)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()

    ax.plot(w_i, L_i)
    # ax.plot([ZPL]*2,[0,max(L_i)],'--k')

    ax.set_xlim(ZPL / 2, ZPL * 1.2)
    ax.set_ylim(0, max(L_i) * 1.1)
    ax.set_xlabel(r'Energy (eV)', size=16)
    ax.set_ylabel('PL (a.u.)', size=18)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
