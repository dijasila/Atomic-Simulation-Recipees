import ase.io.ulm as ulm
from ase.calculators.emt import EMT


class KohnShamConvergenceError(Exception):
    ...


class GPAW(EMT):
    def __init__(self, **kwargs):
        EMT.__init__(self)

    def write(self, filename):
        with ulm.open(filename, 'w') as w:
            w.write(hello='world')

