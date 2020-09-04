from asr.core import command, option, read_json
import ase.units as units
from math import sqrt
import numpy as np


@command("asr.config_diagram")
@option('--state', help='Which configuration is displaced.', type=str)
@option('--npoints', help='How many displacement points.', type=int)
def calculate(state: str = 'ground', npoints: int = 5):
    """Calculate the energies of the displaced structures
       along the one-dimensional mode"""

    from gpaw import restart

    if state == 'ground':
        atoms_1, calc_1 = restart('gs.gpw', txt=None)
        atoms_2, calc_2 = restart('excited/gs.gpw', txt=None)

    if state == 'excited':
        atoms_1, calc_1 = restart('gs.gpw', txt=None)
        atoms_2, calc_2 = restart('../gs.gpw', txt=None)

    displ_n = np.linspace(-1.0, 1.0, npoints, endpoint=True)
    m_a = atoms_1.get_masses()
    pos_ai = atoms_1.positions.copy()

    delta_R = atoms_2.positions - atoms_1.positions
    delta_Q = ((delta_R**2).sum(axis=-1) * m_a).sum()
    zpl = abs(atoms_1.get_potential_energy() - atoms_2.get_potential_energy())

    Q_n = []
    energies_n = []

    for displ in displ_n:
        Q2 = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        Q_n.append(sqrt(Q2) * np.sign(displ))

        atoms_1.positions += displ * delta_R
        atoms_1.positions = pos_ai
        atoms_1.set_calculator(calc_1)
        # atoms_1.get_potential_energy()

        energy = 0.06155**2 / 2 * 15.4669**2 * Q2
        energies_n.append(energy)

    results = {'delta_Q': delta_Q,
               'Q_n': Q_n,
               'energies_n': energies_n,
               'ZPL': zpl}

    return results


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Configuration coordinate diagram',
             'columns': [[fig('cc_diagram.png')]],
             'plot_descriptions': [{'function': plot_cc_diagram,
                                    'filenames': ['cc_diagram.png']}],
             'sort': 13}

    return [panel]


@command("asr.config_diagram",
         webpanel=webpanel,
         dependencies=["asr.config_diagram@calculate"])
def main():
    """Estrapolate the frequencies of the ground and
       excited one-dimensional mode and their relative
       Huang-Rhys factors"""

    data_g = read_json('results-asr.config_diagram@calculate.json')
    data_e = read_json('excited/results-asr.config_diagram@calculate.json')
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

    ground = {'energies_n': energies_gn,
              'omega': omega_g,
              'S': S_g}

    excited = {'energies_n': energies_en,
               'omega': omega_e,
               'S': S_e}

    results = {'Q_n': Q_n,
               'ZPL': zpl,
               'delta_Q': delta_Q,
               'ground': ground,
               'excited': excited}

    return results


def plot_cc_diagram(row, fname):
    from matplotlib import pyplot as plt
    from asr.core import read_json

    data = row.data.get('results-asr.config_diagram.json')
    data_g = data['ground']
    data_e = data['excited']

    ene_g = data_g['energies_n']
    ene_e = data_e['energies_n']

    Q_n = np.array(data['Q_n'])
    ZPL = data['ZPL']
    delta_Q = data['delta_Q']**0.5
    q = np.linspace(Q_n[0] - delta_Q * 0.2, Q_n[-1] + delta_Q * 0.2, 100)

    omega_g = data_g['omega']
    omega_e = data_e['omega']

    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()

    # Helping lines
    ax.plot(q, 1 / 2 * omega_g**2 * s**2 * q**2, '-C0')
    ax.plot(Q_n, ene_g, 'wo', ms=7, markeredgecolor='C0', markeredgewidth=0.9)
    # Ground state parabola
    ax.plot(q, 1 / 2 * omega_g**2 * s**2 * q**2, '-C0')
    ax.plot(Q_n, ene_g, 'wo', ms=7, markeredgecolor='C0', markeredgewidth=0.9)
    # Excited state parabola
    ax.plot(q + delta_Q, 1 / 2 * omega_e**2 * s**2 * q**2 + ZPL, '-C1')
    ax.plot(Q_n + delta_Q, ene_e + ZPL, 'wo', ms=7,
            markeredgecolor='C1', markeredgewidth=0.9)

    ax.set_xlabel(r'Q$\;(amu^{1/2}\AA)$', size=14)
    ax.set_ylabel(r'E(Q) $-$ E(0) (eV)', size=14)
    ax.set_xlim(-1.3 * delta_Q, 2 * delta_Q * 1.15)
    ax.set_ylim(-1 / 5 * ZPL, 1.1 * max(ene_e) + 6 / 5 * ZPL)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
