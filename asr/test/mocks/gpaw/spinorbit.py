import numpy as np


def soc_eigenstates(
        calc,
        n1=None,
        n2=None,
        scale=1.0,
        theta=0.0,
        phi=0.0,
        eigenvalues=None,
        occcalc=None,
        projected=None):

    nk = len(calc.get_ibz_k_points())
    nspins = 2
    nbands = calc.get_number_of_bands()
    n1 = n1 or 0
    n2 = n2 or nbands

    e_ksn = np.array(
        [
            [
                calc.get_eigenvalues(kpt=k, spin=s)[n1:n2]
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
    return SOC(e_km, s_kvm, calc.get_fermi_level(), theta)


class SOC:
    def __init__(self, e_km, s_kvm, fermi_level, theta):
        self.eig_km = e_km
        self.s_kmv = s_kvm.transpose((0, 2, 1))
        self.fermi_level = fermi_level
        self.theta = theta

    def eigenvalues(self):
        return self.eig_km

    def spin_projections(self):
        return self.s_kmv

    def calculate_band_energy(self):
        return self.theta
