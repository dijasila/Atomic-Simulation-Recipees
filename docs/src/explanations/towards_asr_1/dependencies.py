from ase.build import bulk

from asr.relax import main as relax
from asr.gs import main as gs

atoms = bulk('Ag')

relaxrecord = relax(atoms, calculator=dict(name='emt'))
relaxed_atoms = relaxrecord.result.atoms

gsrecord = gs(relaxed_atoms,
              calculator=dict(name='gpaw', precision='low', txt=None))
