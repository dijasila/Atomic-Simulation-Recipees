from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac
from ase.build import molecule
from ase.calculators.dftd3 import DFTD3
import numpy as np

atoms = molecule('N2')
atoms.center(vacuum=3.5)
dft = GPAW(xc='PBE', mode=PW(400),
           txt='N2.txt', occupations=FermiDirac(width=0.02))
d3 = DFTD3(dft=dft)
atoms.set_calculator(d3)
relax = BFGS(atoms, logfile='relax.log', trajectory='relax.traj')
relax.run(fmax=0.01)
dft.write('gs.gpw')
edft = dft.get_potential_energy(atoms)
efull = atoms.get_potential_energy()
msg = 'DFT energies not reproducing hardcoded value'
assert np.allclose(edft, -13.2538334148663, 5), msg
msg = 'D3 energies not reproducing hardcoded value'
ed3 = efull - edft
assert np.allclose(ed3, -5.7143910652257546e-06, 8), msg
