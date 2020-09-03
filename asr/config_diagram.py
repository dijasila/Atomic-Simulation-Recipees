from asr.core import command, read_json, write_json


def webpanel(row, key_descriptions):

    panel = {'title': 'Configuration coordinate'}

    return [panel]


@command('asr.config_diagram',
         dependencies=['asr.setup.cc-diagram'],
         webpanel=webpanel)
def calculate():
    """Calculate the energies of the displaced structures
       along the one-dimensional mode"""

    from gpaw import GPAW
    from ase.io import read

    setup = read_json('results-asr.setup.cc-diagram.json')
    folders = setup['folders']
    state = setup['__params__']['state']

    calc = GPAW('gs.gpw', txt=None)
    if state == 'excited':
        calc = GPAW('ex.gpw', txt=None)

    for folder in folders:
        params = read_json(folder + '/params.json')
        Q = params['Q']

        atoms = read(folder + '/structure.json')
        atoms.set_calculator(calc)
        # atoms.get_potential_energy()
        energy = 0.06155**2 / 2 * 15.4669**2 * Q**2
        params['energy'] = energy
        write_json(folder + '/params.json', params)


@command("asr.config_diagram",
         webpanel=webpanel,
         dependencies=["asr.config_diagram@calculate"])
def main():
    """Estrapolate the frequencies of the ground and
       excited one-dimensional mode and their relative
       Huang-Rhys factors"""
    import ase.units as units
    from math import sqrt
    import numpy as np

    setup = read_json('results-asr.setup.cc-diagram.json')
    folders = setup['folders']
    delta_Q = setup['delta_Q']

    energies_n = []
    Q_n = []

    for folder in folders:
        params = read_json(folder + '/params.json')
        Q = params['Q']
        energy = params['energy']
        Q_n.append(Q)
        energies_n.append(energy)

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

    results = {'Q_n': Q_n,
               'energies_n': energies_n,
               'omega': omega,
               'S': S}

    return results


if __name__ == '__main__':
    main.cli()
