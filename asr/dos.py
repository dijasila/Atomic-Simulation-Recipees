"""Density of states."""
from asr.core import command, option, atomsopt, calcopt
from asr.gs import calculate as gscalculate
from ase import Atoms



@command('asr.dos')
@atomsopt
@calcopt
@option('--kptdensity', help='K point kptdensity', type=float)
def main(
        atoms: Atoms,
        calculator: gscalculate.defaults.calculator,
        kptdensity: float = 12.0,
) -> dict:
    """Calculate DOS."""
    from pathlib import Path
    from gpaw import GPAW

    result = gscalculate(atoms=atoms, calculator=calculator)
    name = 'dos.gpw'
    if not Path(name).is_file():
        calc = result.calculation.load(
            kpts=dict(density=kptdensity),
            nbands='300%',
            convergence={'bands': -10},
        )
        calc.get_potential_energy()
        calc.write(name)
        del calc

    calc = GPAW(name, txt=None)

    from ase.dft.dos import DOS
    dos = DOS(calc, width=0.0, window=(-5, 5), npts=1000)
    nspins = calc.get_number_of_spins()
    dosspin0_e = dos.get_dos(spin=0)
    energies_e = dos.get_energies()
    natoms = len(calc.atoms)
    volume = calc.atoms.get_volume()
    data = {'dosspin0_e': dosspin0_e.tolist(),
            'energies_e': energies_e.tolist(),
            'natoms': natoms,
            'volume': volume}
    if nspins == 2:
        dosspin1_e = dos.get_dos(spin=1)
        data['dosspin1_e'] = dosspin1_e.tolist()

<<<<<<< HEAD
    import json
 
    from ase.parallel import paropen
    with paropen(filename, 'w') as fd:
        json.dump(data, fd)


def collect_data(atoms):
    """Band structure PBE and GW +- SOC."""
    from ase.io.jsonio import read_json
    from pathlib import Path

    if not Path('dos.json').is_file():
        return {}, {}, {}

    dos = read_json('dos.json')

    return {}, {}, {'dos': dos}
=======
    return data
>>>>>>> origin/master


def plot(row=None, filename='dos.png', file=None, show=False):
    """Plot DOS.

    Defaults to dos.json.
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    dos = None

    # Get data from row
    if row is not None:
        if 'results-asr.dos.json' not in row.data:
            return
        dos = row.data['results-asr.dos.json']

    # Otherwise from from file
    file = 'results-asr.dos.json'
    if not dos:
        dos = json.load(open(file, 'r'))
    plt.figure()
    plt.plot(dos['energies_e'],
             np.array(dos['dosspin0_e']) / dos['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.tight_layout()
    plt.savefig(fname)
    if show:
        plt.show()
    return plt.gca()


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = (f'Density of states ({xcname})',
             [[fig('dos.png')], []])

    things = [(plot, ['dos.png'])]

    return panel, things


if __name__ == '__main__':
    main.cli()

