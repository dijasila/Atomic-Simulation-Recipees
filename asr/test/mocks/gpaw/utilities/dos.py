import numpy as np


def raw_orbital_LDOS(calc, a, spin, angular, nbands=None):
    eigenvalues = calc.eigenvalues
    weights_xi = np.zeros(list(eigenvalues.shape) + [len(angular)]) + 1 / len(angular)
    return eigenvalues[:nbands], weights_xi
