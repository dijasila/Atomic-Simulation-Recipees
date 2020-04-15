import numpy as np
from gpaw import GPAW
from ase.units import Ha


def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
    fermi_level = GPAW._fermi_level / Ha  # This is a hack
    f_skn = (eps_skn < 0.0).astype(float)
    f_skn /= np.prod(f_skn.shape) * nelectrons
    return f_skn, fermi_level, 0.0, 0.0
