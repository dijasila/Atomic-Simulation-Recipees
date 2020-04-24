import numpy as np
from ase.units import Ha


def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
    from gpaw import GPAW
    fermi_level = GPAW._fermi_level / Ha  # This is a hack
    f_skn = (eps_skn < 0.0).astype(float)
    f_skn /= np.prod(f_skn.shape) * nelectrons
    return f_skn, fermi_level, 0.0, 0.0


class FermiDirac:

    def __init__(self, *args, **kwargs):
        self = None
