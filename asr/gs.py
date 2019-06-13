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


@command('asr.gs', defaults,
         creates=['gs.gpw'])
@option('-a', '--atomfile', type=str,
        help='Atomic structure',
        default='structure.json')
@option('--ecut', type=float, help='Plane-wave cutoff', default=800)
@option(
    '-k', '--kptdensity', type=float, help='K-point density', default=6.0)
@option('--xc', type=str, help='XC-functional', default='PBE')
@option('--width', default=0.05,
        help='Fermi-Dirac smearing temperature')
def main(atomfile, ecut, xc, kptdensity, width):
    """Calculate ground state density.

    By default, this recipe reads the structure in 'structure.json'
    and saves a gs.gpw file containing the ground state density."""
    from ase.io import read
    from asr.calculators import get_calculator
    atoms = read('structure.json')

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

    calc = get_calculator()(**params)

    atoms.calc = calc
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw')


def postprocessing():
    """Extract data from groundstate in gs.gpw.

    This function is very fast which can be nice when our
    main function can be divided into a fast and a slow part.
    In this case, the slow part is the actual calculation of the
    ground state file 'gs.gpw' and the fast part is simply extracting
    the relevant results from this file."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)
    forces = calc.get_forces()
    stresses = calc.get_stress()
    etot = calc.get_potential_energy()
    
    fingerprint = {}
    for setup in calc.setups:
        fingerprint[setup.symbol] = setup.fingerprint

    results = {'forces': forces,
               'stresses': stresses,
               'etot': etot,
               '__key_descriptions__':
               {'forces': 'Forces on atoms [eV/Angstrom]',
                'stresses': 'Stress on unit cell [eV/Angstrom^dim]',
                'etot': 'KVP: Total energy [eV]'},
               '__setup_fingerprints__': fingerprint}
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
