"""Density of states."""
from __future__ import annotations
from asr.core import command, option, ASRResult, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  make_panel_description, fig)


panel_description = make_panel_description(
    """DOS
...""")


def webpanel(result, row, key_descriptions):
    parameter_description = entry_parameter_description(
        row.data,
        'asr.dos')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Effective masses',
                                     description=title_description),
             'columns': [[fig('dos.png')]],
             'plot_descriptions':
                 [{'function': dos_plot,
                   'filenames': ['dos.png']}]}

    return [panel]


@prepare_result
class DOSResult(ASRResult):
    dosspin0_e: list[float]
    dosspin1_e: list[float]
    energies_e: list[float]
    natoms: int
    volume: float

    key_descriptions = {'dosspin0_e': '...',
                        'dosspin1_e': '...',
                        'energies_e': '...',
                        'natoms': '...',
                        'volume': '...'}
    formats = {"ase_webpanel": webpanel}


@command('asr.dos',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'])
@option('--name', type=str)
@option('--kptdensity', help='K point kptdensity', type=float)
def main(name: str = 'dos.gpw',
         kptdensity: float = 12.0) -> DOSResult:
    """Calculate DOS."""
    from pathlib import Path
    from gpaw import GPAW
    if not Path(name).is_file():
        calc = GPAW('gs.gpw', txt='dos.txt',
                    kpts={'density': kptdensity},
                    nbands='300%',
                    convergence={'bands': -10})
        calc.get_potential_energy()
        calc.write(name)
        del calc

    calc = GPAW(name)
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
    return DOSResult(data=data)


def dos_plot(row, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    dos = row.data.get('results-asr.dos.json')
    fig, ax = plt.subplots()
    ax.plot(dos['energies_e'],
            np.array(dos['dosspin0_e']) / dos['volume'])
    ax.set_xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    ax.set_ylabel(r'DOS (states / (eV Å$^3$)')
    fig.tight_layout()
    fig.savefig(filename)
    return [ax]
