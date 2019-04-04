import ase.io.ulm as ulm
from ase.calculators.emt import EMT


class KohnShamConvergenceError(Exception):
    ...


class GPAW(EMT):
    def __init__(self, **kwargs):
        EMT.__init__(self)

    def write(self, filename):
        from ase.io.trajectory import write_atoms
        with ulm.open(filename, 'w') as w:
            write_atoms(w.child('atoms'), self.atoms)
            w.child('results').write(**self.results)
            w.child('wave_functions').write(foo='bar')
            w.child('occupations').write(fermilevel=42)
