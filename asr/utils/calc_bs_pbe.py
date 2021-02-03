from ase.build import mx2
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors
from ase.visualize import view
from ase.io import read

a12 = read("structure.json")

k = 20
a12.calc = GPAW(mode='lcao',
                basis='dzp',
                nbands='nao',
                kpts={'size': (k, k, 1), 'gamma': True},
                txt=None)

a12.get_potential_energy()
a12.calc.write('gs_full.gpw')

bp = a12.cell.bandpath('GMKG', npoints=80)

calc = GPAW('gs_lcao.gpw',
            fixdensity=True,
            symmetry='off',
            kpts=bp,
            txt=None)

calc.get_potential_energy()
calc.write('bs_full.gpw', 'all')




