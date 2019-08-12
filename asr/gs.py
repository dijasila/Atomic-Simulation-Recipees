from asr.utils import command, option
from pathlib import Path

# Get some parameters from structure.json
defaults = {}
if Path('results_relax.json').exists():
    from asr.utils import read_json
    dct = read_json('results_relax.json')['__params__']
    if 'ecut' in dct:
        defaults['ecut'] = dct['ecut']

tests = []
tests.append({'description': 'Test ground state of Si.',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params asr.gs:ecut 300 '
                      'asr.gs:kptdensity 2"',
                      'asr run gs',
                      'asr run database.fromtree',
                      'asr run "browser --only-figures"']})


@command(module='asr.gs',
         overwrite_defaults=defaults,
         creates=['gs.gpw'],
         tests=tests,
         dependencies=['asr.structureinfo'],
         resources='8:10h',
         restart=1)
@option('-a', '--atomfile', type=str, help='Atomic structure')
@option('--ecut', type=float, help='Plane-wave cutoff')
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--xc', type=str, help='XC-functional')
@option('--width', help='Fermi-Dirac smearing temperature')
def main(atomfile='structure.json', ecut=800, xc='PBE',
         kptdensity=6.0, width=0.05):
    """Calculate ground state.

    This recipe saves the ground state to a file gs.gpw based on the structure
    in 'structure.json'. This can then be processed by asr.gs@postprocessing
    for storing any derived quantities. See asr.gs@postprocessing for more
    information."""
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
        occupations={'name': 'fermi-dirac', 'width': width},
        txt='gs.txt')

    calc = get_calculator()(**params)

    atoms.calc = calc
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw')


@command(module='asr.gs',
         dependencies=['asr.gs@main'])
def postprocessing():
    """Extract derived quantities from groundstate in gs.gpw."""
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


if __name__ == '__main__':
    postprocessing.cli()
