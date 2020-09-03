from asr.core import command, option, read_json
import numpy as np


def webpanel(row, key_descriptions):

    panel = {'title': 'Configuration coordinate'}

    return [panel]


@command('asr.config_diagram',
         webpanel=webpanel)
@option('--state', help='Which confguration is displaced.', type=str)
@option('--npoints', help='How many displacement points.', type=int)
def calculate(state: str = 'ground', npoints: int = 5):
    """Calculate the energies of the displaced structures
       along the one-dimensional mode"""

    from gpaw import GPAW
    from ase.io import read


    if state == 'excited':
        atoms_1 = read('structure.json')
        atoms_2 = read('../structure.json')
    else:
        atoms_1 = read('structure.json')
        atoms_2 = read('excited/structure.json')
    
    calc = GPAW('gs.gpw', txt=None)

    displ_n = np.linspace(-1.0, 1.0, npoints, endpoint=True)
    m_a = atoms_1.get_masses()
    pos_ai = atoms_1.positions.copy()

    delta_R = atoms_2.positions - atoms_1.positions
    delta_Q = ((delta_R**2).sum(axis=-1) * m_a).sum()

    Q_n = []
    energies_n = []

    for displ in displ_n:
        Q = (((displ * delta_R)**2).sum(axis=-1) * m_a).sum()
        Q_n.append(Q)

        atoms_1.positions += displ * delta_R
        atoms_1.positions = pos_ai
        atoms_1.set_calculator(calc)
        # atoms_1.get_potential_energy()

        energy = 0.06155**2 / 2 * 15.4669**2 * Q**2
        energies_n.append(energy)

    results = {'delta_Q': delta_Q,
               'Q_n': Q_n,
               'energies_n': energies_n}

    return results


@command("asr.config_diagram",
         webpanel=webpanel,
         dependencies=["asr.config_diagram@calculate"])
def main():
    """Estrapolate the frequencies of the ground and
       excited one-dimensional mode and their relative
       Huang-Rhys factors"""
    import ase.units as units
    from math import sqrt

    data = read_json('results-asr.config_diagram@calculate.json')
    delta_Q = data['delta_Q']
    energies_n = data['energies_n']
    Q_n = data['Q_n']

    # Rescale energy by the minimum value
    energies_n = np.array(energies_n)
    energies_n -= np.min(energies_n)

    # Quadratic fit of the parabola
    z = np.polyfit(Q_n, energies_n, 2)

    # Conversion factor
    s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar

    # Estrapolation of the effective frequency
    omega = sqrt(2 * z[0] / s**2)

    # Estrapolation of the Huang-Rhys factor
    S = s**2 * delta_Q**2 * omega / 2

    results = {'omega': omega,
               'S': S}

    return results


if __name__ == '__main__':
    main.cli()
