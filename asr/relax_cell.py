from ase.io import read
from ase.io import Trajectory
from gpaw import GPAW
from ase.optimize import BFGS
from ase.constraints import StrainFilter, ExpCellFilter, FixScaled, FixAtoms
from ase.spacegroup.symmetrize import FixSymmetry
from asr.core import command, option
from pathlib import Path
import os


@command('asr.relax_cell')
@option('-s', '--structure')
@option('--traj')
@option('--rattle')
@option('--restart')
@option('--fmax')
@option('--mask')
@option('--calculator')
def main(
    structure: str = 'initial.json', 
    traj: str = 'relax_cell.traj',
    rattle: bool = True,
    restart: bool = True,
    fmax: float = 0.01,
    mask: list = [1, 1, 0, 0, 0, 1],
    calculator: dict = {'mode': {'name': 'pw', 'ecut': 800},
                        'xc': 'PBE',
                        'kpts': {'density': 6.0, 'gamma': True},
                        'basis': 'dzp',
                        'symmetry': {'symmorphic': False},
                        'convergence': {'forces': 1e-4},
                        'txt': 'relax_cell.txt',
                        'occupations': {'name': 'fermi-dirac',
                                        'width': 0.05},
                        'charge': 0}
):

    if restart and os.path.isfile('relax_cell.traj') and os.path.getsize('relax_cell.traj') > 1000:
        atoms = Trajectory(traj)[-1]
    else:
        atoms = read(structure)
    if rattle:
        atoms.rattle(stdev=0.001)
    
    atoms.calc = GPAW(**calculator)
    atoms.set_constraint(FixAtoms(mask = [True for atom in atoms]))
    ecf = ExpCellFilter(atoms, mask=mask, hydrostatic_strain=True)
    relax = BFGS(ecf)
    traj = Trajectory(traj, 'a', atoms)
    relax.attach(traj)
    relax.run(fmax=fmax)
    final = Trajectory('relax_cell.traj')[-1]
    final.set_constraint(None)
    final.write('unrelaxed.json')


if __name__ == '__main__':
    main.cli()
