import numpy as np


def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
    f_skn = (eps_skn < 0.0).astype(float)
    f_skn /= np.prod(f_skn.shape) * nelectrons
    return f_skn, 0.0, 0.0, 0.0
