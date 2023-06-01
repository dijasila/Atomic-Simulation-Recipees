"""Density of states."""
from pathlib import Path
from typing import List

import numpy as np

from asr.core import ASRResult, command, option, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)

panel_description = make_panel_description(
    """Density of States
""")


def webpanel(result: ASRResult,
             row,
             key_descriptions: dict) -> list:
    parameter_description = entry_parameter_description(
        row.data,
        'asr.dos')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Density of States',
                                     description=title_description),
             'columns': [[fig('dos.png')]],
             'plot_descriptions':
                 [{'function': dos_plot,
                   'filenames': ['dos.png']}]}

    return [panel]


@prepare_result
class DOSResult(ASRResult):
    dosspin0_e: List[float]
    dosspin1_e: List[float]
    energies_e: List[float]
    natoms: int
    volume: float

    key_descriptions = {'dosspin0_e': 'Spin up DOS [states/eV]',
                        'dosspin1_e': 'Spin up DOS [states/eV]',
                        'energies_e': 'Energies relative to Fermi level [eV]',
                        'natoms': 'Number of atoms',
                        'volume': 'Volume of unit cell [Ang^3]'}
    formats = {"ase_webpanel": webpanel}


Result = DOSResult  # backwards compatibility with old result files


@command('asr.dos',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'])
@option('--name', type=str)
@option('--kptdensity', help='K-point density', type=float)
def main(name: str = 'dos.gpw',
         kptdensity: float = 12.0) -> ASRResult:
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


def _main(doscalc) -> dict:
    energies_e = np.linspace(-10, 10, 201)
    data = {'energies_e': energies_e.tolist(),
            'dosspin1_e': []}
    for spin in range(doscalc.nspins):
        dos_e = doscalc.raw_dos(energies_e, spin, width=0)
        data[f'dosspin{spin}_e'] = dos_e.tolist()
    return data


def dos_plot(row, filename: str):
    import matplotlib.pyplot as plt
    dos = row.data.get('results-asr.dos.json')
    x = dos['energies_e']
    y0 = dos['dosspin0_e']
    y1 = dos['dosspin1_e']
    fig, ax = plt.subplots()
    if y1:
        ax.plot(x, y0, label='up')
        ax.plot(x, y1, label='down')
        ax.legend()
    else:
        ax.plot(x, y0)

    ax.set_xlabel(r'Energy - $E_\mathrm{F}$ [eV]')
    ax.set_ylabel('DOS [electrons/eV]')
    fig.tight_layout()
    fig.savefig(filename)
    return [ax]
