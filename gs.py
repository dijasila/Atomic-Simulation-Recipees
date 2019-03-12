# Example of recipe. Only necessary keys are "parser"
# which also includes the description

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import json


def main(args):
    from ase.io import read
    from gpaw import GPAW, PW, FermiDirac
    name = args['atoms']
    gpwfilename = args['gpw']
    ecut = args['ecut']
    xc = args['xc']
    kptdens = args['kptdensity']
    params = dict(
        mode=PW(ecut),
        xc=xc,
        basis='dzp',
        kpts={'density': kptdens, 'gamma': True},
        occupations=FermiDirac(width=0.05),
        txt='gs.txt')

    slab = read(name)
    slab.calc = GPAW(**params)
    slab.get_forces()
    slab.get_stress()
    slab.calc.write(gpwfilename)


# The metadata is put it the bottom
group = 'Property'
short_description = 'Calculate ground state density and save to gs.gpw'
description = ''
dependencies = []  # What other recipes does this recipe depend on
creates = ['gs.gpw']  # What files are created
resources = '8:1h'  # How many resources are used
diskspace = 0  # How much diskspace is used
restart = 1  # Does it make sense to restart the script?

# Default parameters
params = {'atoms': 'start.traj',
          'gpw': 'gs.gpw',
          'ecut': 800,
          'kptdensity': 6.0,
          'xc': 'PBE'}

# Load parameters from params.json
if Path('params.json').is_file():
    otherparams = json.load(open('params.json', 'r'))['rmr.gs']
    params.update(otherparams)

# Make parser
parser = ArgumentParser(description=description,
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-a', '--atoms', type=str, default=params['atoms'],
                    help='Atomic structure')
parser.add_argument('-g', '--gpw', type=str, default=params['gpw'],
                    help='Name of ground state file')
parser.add_argument('-e', '--ecut', type=float, default=params['ecut'],
                    help='Plane-wave cutoff')
parser.add_argument('-k', '--kptdensity', type=float,
                    default=params['kptdensity'],
                    help='K-point density')
parser.add_argument('--xc', type=str, default=params['xc'],
                    help='XC-functional')


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
