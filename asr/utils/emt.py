from ase.calculators.emt import EMT


class KohnShamConvergenceError(Exception):
    ...


class GPAW(EMT):
    implemented_properties = ['energy', 'forces', 'stress', 'magmom']

    def __init__(self, **kwargs):
        EMT.__init__(self)
