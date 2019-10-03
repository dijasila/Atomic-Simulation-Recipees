from asr.core import command, option


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig
    print('In dos webpanel')
    panel = {'title': 'Density of states (PBE)',
             'columns': [[fig('dos.png')]],
             'plot_descriptions': [{'function': plot,
                                    'filenames': ['dos.png']}]}

    return [panel]


@command('asr.dos',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         webpanel=webpanel)
@option('--name', type=str)
@option('--filename', type=str)
@option('--kptdensity', help='K point kptdensity')
def main(name='dos50.gpw', filename='dos50.json', kptdensity=50.0):
    """Calculate DOS"""
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
    data = {'dosspin0_e': dosspin0_e.tolist(),
            'energies_e': energies_e.tolist(),
            'natoms': natoms,
            'volume': volume}
    if nspins == 2:
        dosspin1_e = dos.get_dos(spin=1)
        data['dosspin1_e'] = dosspin1_e.tolist()

    return data

    import json
    from ase.parallel import paropen
    with paropen(filename, 'w') as fd:
        json.dump(data, fd)


def plot(row=None, filename='dos.png', file=None, show=False):
    """Plot DOS.

    Defaults to dos.json"""
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    print('In dos plot')
    dos = None
        
    # Get data from row
    if row is not None:
        if 'results-asr.dos.json' not in row.data:
            return
        dos = row.data['results-asr.dos.json']
     
    # Otherwise from from file
    file = file or 'results-asr.dos.json'
    if not dos:
        dos = json.load(open(file, 'r'))
    plt.figure()
    plt.plot(dos['energies_e'],
             np.array(dos['dosspin0_e']) / dos['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.tight_layout()
    #print('saving to file', filename)
    plt.savefig(filename)
    if show:
        plt.show()
    return plt.gca()


def webpanel(row, key_descriptions):
    from asr.browser import fig

    panel = ('Density of states (PBE)',
             [[fig('dos.png')], []])

    things = [(plot, ['dos.png'])]

    return panel, things


group = 'property'
dependencies = ['asr.structureinfo', 'asr.gs']
creates = ['dos.json']

if __name__ == '__main__':
    main.cli()
