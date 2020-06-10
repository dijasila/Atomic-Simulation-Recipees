import numpy as np


def soc_eigenstatest(
        calc,
        bands=None,
        myeig_skn=None,
        scale=1.0,
        theta=0.0,
        phi=0.0,
        return_wfs=False,
        occupations=None):

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
    e_km.sort(-1)  # Make sure eigenvalues are in ascending order
    return {'eigenvalues': e_km,
            'spin_projections': s_kvm}


def get_anisotropy(*args, **kwargs):
    return 0
