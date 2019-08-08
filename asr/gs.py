from asr.utils import command, option
from pathlib import Path

# Get some parameters from structure.json
defaults = {}
if Path('results_relax.json').exists():
    from asr.utils import read_json
    dct = read_json('results_relax.json')['__params__']
    if 'ecut' in dct:
        defaults['ecut'] = dct['ecut']


@command('asr.gs', overwrite_defaults=defaults,
         creates=['gs.gpw'])
@option('-a', '--atomfile', type=str, help='Atomic structure')
@option('--ecut', type=float, help='Plane-wave cutoff')
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--xc', type=str, help='XC-functional')
@option('--width', help='Fermi-Dirac smearing temperature')
def main(atomfile='structure.json', ecut=800, xc='PBE',
         kptdensity=6.0, width=0.05):
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

    This will be called after main by default."""
    from asr.calculators import get_calculator
    calc = get_calculator()('gs.gpw', txt=None)
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
                'etot': 'Total energy [eV]'},
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

tests = []
tests.append({'description': 'Test ground state of Si.',
              'cli': ['asr run setup.materials -s Si2',
                      'ase convert materials.json structure.json',
                      'asr run setup.params asr.gs:ecut 300 '
                      'asr.gs:kptdensity 2',
                      'asr run gs',
                      'asr run database.fromtree',
                      'asr run browser --only-figures']})


if __name__ == '__main__':
    main.cli()
