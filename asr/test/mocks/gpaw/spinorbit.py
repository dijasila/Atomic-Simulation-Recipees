import numpy as np


def get_spinorbit_eigenvalues(
        calc,
        bands=None,
        gw_kn=None,
        return_spin=False,
        return_wfs=False,
        scale=1.0,
        theta=0.0,
        phi=0.0,
):

    nk = len(calc.get_ibz_k_points())
    nspins = 2
    nbands = calc.get_number_of_bands()
    bands = list(range(nbands))

    e_ksn = np.array(
        [
            [
                calc.get_eigenvalues(kpt=k, spin=s)[bands]
                for s in range(nspins)
            ]
            for k in range(nk)
        ]
    )

    s_kvm = np.zeros((nk, 3, nbands * 2), float)
    s_kvm[:, 2, ::2] = 1
    s_kvm[:, 2, ::2] = -1
    e_km = e_ksn.reshape((nk, -1))
    if return_spin:
        return e_km.T, s_kvm
    else:
        return e_km.T


def get_anisotropy(*args, **kwargs):
    return 0
