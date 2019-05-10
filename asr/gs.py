from asr.utils import command, option, get_start_parameters

# Get some parameters from start.json
params = get_start_parameters()
defaults = {}
if 'ecut' in params.get('mode', {}):
    defaults['ecut'] = params['mode']['ecut']

if 'density' in params.get('kpts', {}):
    defaults['kptdensity'] = params['kpts']['density']


@command('asr.gs', defaults)
@option('-a', '--atomfile', type=str,
        help='Atomic structure',
        default='start.json')
@option('--gpwfilename', type=str, help='filename.gpw', default='gs.gpw')
@option('--ecut', type=float, help='Plane-wave cutoff', default=800)
@option(
    '-k', '--kptdensity', type=float, help='K-point density', default=6.0)
@option('--xc', type=str, help='XC-functional', default='PBE')
def main(atomfile, gpwfilename, ecut, xc, kptdensity):
    """Calculate ground state density"""
    from pathlib import Path
    from ase.io import read
    from asr.utils.gpaw import GPAW
    path = Path(atomfile)
    if not path.is_file():
        from asr.utils import get_start_atoms
        atoms = get_start_atoms()
    else:
        atoms = read(atomfile)

    params = dict(
        mode={'name': 'pw', 'ecut': ecut},
        xc=xc,
        basis='dzp',
        kpts={
            'density': kptdensity,
            'gamma': True
        },
        occupations={'name': 'fermi-dirac', 'width': 0.05},
        txt='gs.txt')

    atoms.calc = GPAW(**params)
    atoms.get_forces()
    atoms.get_stress()
    atoms.calc.write(gpwfilename)


# The metadata is put it the bottom
group = 'Property'
description = ''
dependencies = []  # What other recipes does this recipe depend on
creates = ['gs.gpw']  # What files are created
resources = '8:10h'  # How many resources are used
diskspace = 0  # How much diskspace is used
restart = 1  # Does it make sense to restart the script?

if __name__ == '__main__':
    main()
