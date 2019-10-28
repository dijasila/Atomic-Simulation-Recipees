import numpy as np


def get_spin_direction(fname='anisotropy_xy.npz'):
    import os.path as op
    theta = 0
    phi = 0
    if op.isfile(fname):
        data = np.load(fname)
        DE = max(data['dE_zx'], data['dE_zy'])
        if DE > 0:
            theta = np.pi / 2
            if data['dE_zy'] > data['dE_zx']:
                phi = np.pi / 2
    return theta, phi


def spin_axis(fname='anisotropy_xy.npz'):
    theta, phi = get_spin_direction(fname=fname)
    if theta == 0:
        return 2
    elif np.allclose(phi, np.pi / 2):
        return 1
    else:
        return 0
