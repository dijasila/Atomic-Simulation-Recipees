"""Density of states."""
from asr.core import command, option, atomsopt, calcopt
from asr.c2db.gs import calculate as gscalculate
from ase import Atoms


@command('asr.c2db.dos')
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

    return data


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
        if 'dos' not in row.data:
            return
        dos = row.data['dos']

    # Otherwise from from file
    file = file or 'dos.json'
    if not dos:
        dos = json.load(open(file, 'r'))
    plt.figure()
    plt.plot(dos['energies_e'],
             np.array(dos['dosspin0_e']) / dos['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    return plt.gca()


def webpanel(result, context):
    from asr.database.browser import fig

    panel = (f'Density of states ({context.xcname})',
             [[fig('dos.png')], []])

    things = [(plot, ['dos.png'])]

    return panel, things


if __name__ == '__main__':
    main.cli()
