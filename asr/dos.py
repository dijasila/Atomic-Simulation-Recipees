"""Density of states."""
from asr.core import command, option, ASRResult
from asr.database.browser import fig, make_panel_description, describe_entry
from asr.utils.hacks import gs_xcname_from_row

panel_description = make_panel_description(
   """DOS""")

def webpanel(result, row, key_descriptions):
    from asr.database.browser import WebPanel

    xcname = gs_xcname_from_row(row)
    panel = WebPanel(describe_entry(f'Density of states ({xcname})', panel_description),
            columns=[[fig('dos.png')], []],
            plot_descriptions=[{'function': plot,
                                'filenames': ['dos.png']}],
            sort=4)

    return [panel]


@command('asr.dos')
@option('--name', type=str)
@option('--filename', type=str)
@option('--kptdensity', help='K point kptdensity', type=float)
def main(name: str = 'dos.gpw', filename: str = 'dos.json',
         kptdensity: float = 50.0):# -> ASRResult:
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
        results['dosspin1_e'] = dosspin1_e.tolist()

    return Result(data=results)
    


class Result(ASRResult):
    results:dict

    key_descriptions = {"Density of states" : "Density of states"}
    formats = {"ase_webpanel": webpanel}


def plot(row,fname):
    """Plot DOS.

    Defaults to dos.json.
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np


    dos= row.data.get('results-asr.dos.json')

    plt.figure()
    plt.plot(dos['energies_e'],
             np.array(dos['dosspin0_e']) / dos['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
