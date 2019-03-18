# Example of recipe. Only necessary keys are "parser"
# which also includes the description

from asr.utils import get_parser, get_parameters, set_defaults


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
params.update(get_parameters('asr.gs'))

# Make parser
parser = get_parser(description)
parser.add_argument('-a', '--atoms', type=str, help='Atomic structure')
parser.add_argument('-g', '--gpw', type=str, help='Name of ground state file')
parser.add_argument('-e', '--ecut', type=float, help='Plane-wave cutoff')
parser.add_argument('-k', '--kptdensity', type=float, help='K-point density')
parser.add_argument('--xc', type=str, help='XC-functional')
set_defaults(parser, params)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
