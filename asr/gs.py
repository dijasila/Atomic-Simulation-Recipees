from asr.utils import command, option
from pathlib import Path

# Get some parameters from structure.json
defaults = {}
if Path('gs_params.json').exists():
    from asr.utils import read_json
    dct = read_json('gs_params.json')
    if 'ecut' in dct.get('mode', {}):
        defaults['ecut'] = dct['mode']['ecut']

    if 'density' in dct.get('kpts', {}):
        defaults['kptdensity'] = dct['kpts']['density']


@command('asr.gs', defaults)
@option('-a', '--atomfile', type=str,
        help='Atomic structure',
        default='structure.json')
@option('--gpwfilename', type=str, help='filename.gpw', default='gs.gpw')
@option('--ecut', type=float, help='Plane-wave cutoff', default=800)
@option(
    '-k', '--kptdensity', type=float, help='K-point density', default=6.0)
@option('--xc', type=str, help='XC-functional', default='PBE')
@option('--width', default=0.05,
        help='Fermi-Dirac smearing temperature')
def main(atomfile, gpwfilename, ecut, xc, kptdensity, width):
    """Calculate ground state density"""
    from ase.io import read
    from asr.calculators.gpaw import GPAW
    atoms = read(atomfile)

    if Path('gs.gpw').is_file():
        calc = GPAW('gs.gpw', txt=None)
    else:
        params = dict(
            mode={'name': 'pw', 'ecut': ecut},
            xc=xc,
            basis='dzp',
            kpts={
                'density': kptdensity,
                'gamma': True
            },
            symmetry={'do_not_symmetrize_the_density': True},
            occupations={'name': 'fermi-dirac', 'width': width},
            txt='gs.txt')

        calc = GPAW(**params)

    atoms.calc = calc
    forces = atoms.get_forces()
    stresses = atoms.get_stress()
    etot = atoms.get_potential_energy()
    atoms.calc.write(gpwfilename)

    results = {'forces': forces,
               'stresses': stresses,
               'etot': etot}
    return results


# The metadata is put it the bottom
group = 'property'
description = ''
dependencies = ['asr.structureinfo']
creates = ['gs.gpw']
resources = '8:10h'
diskspace = 0
restart = 1

if __name__ == '__main__':
    main()
