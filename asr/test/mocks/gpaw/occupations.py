import numpy as np
from ase.units import Ha
from gpaw import GPAW


def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
    fermilevel = GPAW.default_parameters.get("fermi_level") / Ha
    f_skn = (eps_skn < fermilevel).astype(float)
    f_skn /= np.prod(f_skn.shape) * nelectrons
    return f_skn, fermilevel, 0.0, 0.0
