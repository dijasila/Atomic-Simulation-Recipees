from ase.build import mx2
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors
from ase.visualize import view
from ase.io import read

atoms = read("structure.json")

atoms.calc = GPAW(mode='lcao',
                basis='dzp',
                nbands='nao',
                kpts={'density': 12.0, 'gamma': True},
                occupations={'name': 'fermi-dirac',
                             'width': 0.05},
                txt='gs_lcao.gpw')

atoms.get_potential_energy()
atoms.calc.write('gs_lcao.gpw')

bp = atoms.cell.bandpath(npoints=160, pbc=atoms.pbc, eps=1e-2)

calc = GPAW('gs_lcao.gpw',
            fixdensity=True,
            symmetry='off',
            kpts=bp,
            txt='bs_lcao.gpw')

calc.get_potential_energy()
calc.write('bs_lcao.gpw', 'all')




