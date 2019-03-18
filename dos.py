import argparse


def calculate(name='dos.gpw'):
    from pathlib import Path
    from gpaw import GPAW
    if not Path(name).is_file():
        calc = GPAW('gs.gpw', txt='dos.txt',
                    kpts={'density': 12.0},
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
    filename = 'dos.json'
    
    from ase.parallel import paropen
    with paropen(filename, 'w') as fd:
        json.dump(data, fd)


def plot(files):
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    
    for file in files:
        dct = json.load(open(file, 'r'))
        plt.plot(dct['energies_e'],
                 np.array(dct['dosspin0_e']) / dct['volume'])
    plt.xlabel(r'Energy - $E_\mathrm{F}$ (eV)')
    plt.ylabel(r'DOS (states / (eV Å$^3$)')
    plt.show()


def main(args=None):
    if args['command'] == 'plot':
        plot(args['filenames'])
    else:
        calculate(args['name'])


short_description = 'Calculate density of states'
parser = argparse.ArgumentParser(description=short_description)
help = 'Name of calculator to calculate DOS from'
parser.add_argument('-n', '--name', type=str,
                    default='dos.gpw',
                    help=help)
subparsers = parser.add_subparsers(dest='command')
sparser = subparsers.add_parser('plot', help='plot')
sparser.add_argument('filenames', help='DOS files',
                     default=['dos.json'],
                     metavar='dosfile',
                     nargs='*')

dependencies = ['asr.gs']


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
