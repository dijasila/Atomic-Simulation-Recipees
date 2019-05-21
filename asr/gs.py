from asr.utils import command, option, get_start_parameters

# Get some parameters from structure.json
params = get_start_parameters()
defaults = {}
if 'ecut' in params.get('mode', {}):
    defaults['ecut'] = params['mode']['ecut']

if 'density' in params.get('kpts', {}):
    defaults['kptdensity'] = params['kpts']['density']


@command('asr.gs', defaults)
@option('-a', '--atomfile', type=str,
        help='Atomic structure',
        default='structure.json')
@option('--gpwfilename', type=str, help='filename.gpw', default='gs.gpw')
@option('--ecut', type=float, help='Plane-wave cutoff', default=800)
@option(
    '-k', '--kptdensity', type=float, help='K-point density', default=6.0)
@option('--xc', type=str, help='XC-functional', default='PBE')
def main(atomfile, gpwfilename, ecut, xc, kptdensity):
    """Calculate ground state density"""
    from ase.io import read
    from asr.utils.gpaw import GPAW
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
group = 'property'
description = ''
dependencies = ['asr.quickinfo']
creates = ['gs.gpw']
resources = '8:10h'
diskspace = 0
restart = 1

if __name__ == '__main__':
    main()
