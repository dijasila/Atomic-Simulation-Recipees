from asr.core import command, option

@command('asr.bandstructure',
         requires=['gs.gpw'],
         creates=['bs.gpw'])
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints', type=int)
@option('--emptybands', type=int)
def calculate(kptpath: Union[str, None] = None, npoints: int = 400,
              emptybands: int = 20)
    """Calculate electronic band structure."""
    from gpaw import GPAW
    from ase.io import read
    atoms = read('structure.json')
    if kptpath is None:
        path = atoms.cell.bandpath(npoints=npoints, pbc=atoms.pbc)
    else:
        path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                                   pbc=atoms.pbc)

    convbands = emptybands // 2
    parms = {
        'basis': 'dzp',
        'nbands': -emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'kpts': path,
        'convergence': {
            'bands': -convbands},
        'symmetry': 'off'}
    calc = GPAW('gs.gpw', **parms)
    calc.get_potential_energy()
    calc.write('bs.gpw')

