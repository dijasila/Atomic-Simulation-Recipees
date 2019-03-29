from asr.utils import update_defaults
from functools import partial
import click
click.option = partial(click.option, show_default=True)


@click.command()
@update_defaults('asr.dos')
@click.option('--name', default='dos.gpw', type=str)
@click.option('--filename', default='dos.json', type=str)
@click.option('--density', default=12.0, help='K point density')
def main(name, filename, density):
    """Calculate DOS"""
    from pathlib import Path
    from gpaw import GPAW
    if not Path(name).is_file():
        calc = GPAW('gs.gpw', txt='dos.txt',
                    kpts={'density': density},
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
    data = {'dosspin0_e': dosspin0_e.tolist(),
            'energies_e': energies_e.tolist(),
            'natoms': natoms,
            'volume': volume}
    if nspins == 2:
        dosspin1_e = dos.get_dos(spin=1)
        data['dosspin1_e'] = dosspin1_e.tolist()

    import json
    
    from ase.parallel import paropen
    with paropen(filename, 'w') as fd:
        json.dump(data, fd)


@click.command()
@click.argument('files', type=str, nargs=-1)
def plot(files):
    """Plot DOS.

    Defaults to dos.json"""
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    if not files:
        files = ['dos.json']
    for file in files:
        dct = json.load(open(file, 'r'))
        plt.plot(dct['energies_e'],
                 np.array(dct['dosspin0_e']) / dct['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.show()


group = 'Property'
dependencies = ['asr.gs']
creates = ['dos.json']

if __name__ == '__main__':
    main()
