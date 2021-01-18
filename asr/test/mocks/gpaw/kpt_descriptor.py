import numpy as np


def to1bz(bzk_kc, cell_cv):
    """Wrap k-points to 1. BZ.

    Return k-points wrapped to the 1. BZ.

    bzk_kc: (n,3) ndarray
        Array of k-points in units of the reciprocal lattice vectors.
    cell_cv: (3,3) ndarray
        Unit cell.
    """
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    K_kv = np.dot(bzk_kc, B_cv)
    N_xc = np.indices((3, 3, 3)).reshape((3, 27)).T - 1
    G_xv = np.dot(N_xc, B_cv)

    bz1k_kc = bzk_kc.copy()

    # Find the closest reciprocal lattice vector:
    for k, K_v in enumerate(K_kv):
        # If a k-point has the same distance to several reciprocal
        # lattice vectors, we don't want to pick a random one on the
        # basis of numerical noise, so we round off the differences
        # between the shortest distances to 6 decimals and chose the
        # one with the lowest index.
        d = ((G_xv - K_v)**2).sum(1)
        x = (d - d.min()).round(6).argmin()
        bz1k_kc[k] -= N_xc[x]

    return bz1k_kc


def kpts2sizeandoffsets(*args, **kwargs):
    return [6, 6, 6], [0, 0, 0]
