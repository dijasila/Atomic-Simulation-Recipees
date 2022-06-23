"""Density of states."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from asr.core import ASRResult, command, option, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)

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
@option('--kptdensity', help='K-point density', type=float)
def main(name: str = 'dos.gpw',
         kptdensity: float = 12.0) -> DOSResult:
    """Calculate DOS."""
    from gpaw import GPAW

    path = Path(name)
    if not path.is_file():
        calc = GPAW(path.with_name('gs.gpw'),
                    txt=path.with_name('dos.txt')).fixed_density(
            kpts={'density': kptdensity},
            nbands='300%',
            convergence={'bands': -10})
        calc.write(path)

    calc = GPAW(path)
    doscalc = calc.dos()
    data = _main(doscalc)
    data['natoms'] = len(calc.atoms)
    data['volume'] = calc.atoms.get_volume()
    return DOSResult(data=data)


def _main(doscalc):
    energies_e = np.linspace(-10, 10, 201)
    data = {'energies_e': energies_e.tolist(),
            'dosspin1_e': []}
    for spin in range(doscalc.nspins):
        dos_e = doscalc.raw_dos(energies_e, spin, width=0)
        data[f'dosspin{spin}_e'] = dos_e.tolist()
    return data


def dos_plot(row, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    dos = row.data.get('results-asr.dos.json')
    fig, ax = plt.subplots()
    dos_e = np.array(dos['dosspin0_e'])
    if 'dosspin1_e' in dos:
        dos_e += dos['dosspin0_e']
    else:
        dos_e *= 2

    ax.plot(dos['energies_e'], dos_e / dos['volume'])
    ax.set_xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    ax.set_ylabel(r'DOS (states / (eV Å$^3$)')
    fig.tight_layout()
    fig.savefig(filename)
    return [ax]
