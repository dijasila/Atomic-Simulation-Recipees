"""Density of states."""
from asr.core import command, option, ASRResult


@command('asr.dos')
@option('--name', type=str)
@option('--filename', type=str)
@option('--kptdensity', help='K point kptdensity', type=float)
def main(name: str = 'dos.gpw', filename: str = 'dos.json',
         kptdensity: float = 12.0) -> ASRResult:
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

    calc = GPAW(name, txt=None)
    from ase.dft.dos import DOS
    dos = DOS(calc, width=0.0, window=(-5, 5), npts=1000)
    nspins = calc.get_number_of_spins()
    dosspin0_e = dos.get_dos(spin=0)
    energies_e = dos.get_energies()
    natoms = len(calc.atoms)
    volume = calc.atoms.get_volume()
    results= {'dosspin0_e': dosspin0_e.tolist(),
            'energies_e': energies_e.tolist(),
            'natoms': natoms,
            'volume': volume}
    if nspins == 2:
        dosspin1_e = dos.get_dos(spin=1)
        data['dosspin1_e'] = dosspin1_e.tolist()


    return results


def plot(row,filename):
    """Plot DOS.

    Defaults to dos.json.
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np


    print('heeeeeej')
    dos= row.data.get('results-asr.dos.json')

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


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig
    from asr.utils.hacks import gs_xcname_from_row
    from asr.database.browser import WebPanel
    xcname = gs_xcname_from_row(row)
    print('heeeej')
    panel = WebPanel(title=f'Density of states ({xcname})',
            columns=[[fig('dos.png')], []],
            plot_descriptions=[{'function': plot,
                                'filenames': ['dos.png']}],
            sort=3)

    return [panel]


if __name__ == '__main__':
    main.cli()
