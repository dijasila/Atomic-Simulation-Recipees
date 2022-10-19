import ase.units as units
from math import sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
from asr.core import write_json


class Lineshape:

    def __init__(self, ground, excited, phonon,
                 sigma=0.01, delta_t=0.01, N=18,
                 gamma=0.05):

        self.ground = ground
        self.excited = excited
        self.phonon = phonon
        self.masses_a = ground.get_masses()
        self.sigma = sigma
        self.gamma = gamma

        self.N_max = 2**N
        self.omega_k = np.arange(0, self.N_max) * (2 * pi / (self.N_max * delta_t))
        print(self.omega_k[0], self.omega_k[-1])

    def broadening(self, omega_i, omega_0, sigma):
        gauss_i = 1 / (sigma * sqrt(2 * pi)) \
            * np.exp(-1 / 2 * ((omega_i - omega_0) / sigma)**2)

        return gauss_i

    def get_delta_Q(self):
        delta_R_av = self.excited.positions - self.ground.positions
        delta_Q = sqrt(((delta_R_av**2).sum(axis=-1) * self.masses_a).sum())

        return delta_Q, delta_R_av

    def get_eigenvalues_and_eigenvectors(self):
        """Get eigenvalues and eigenvectors from a phonon object."""
        atoms = self.ground
        omega_l, modes_ll = self.phonon.get_frequencies_with_eigenvectors([0, 0, 0])
        omega_l = omega_l * 4.1357 / 1000
        modes_lav = modes_ll.T.reshape(3 * len(atoms), len(atoms), 3)

        return omega_l, modes_lav

    def get_partial_hr(self):
        s = np.sqrt(units._e * units._amu) * 1e-10 / units._hbar
        omega_l, modes_lav = self.get_eigenvalues_and_eigenvectors()
        delta_Q, delta_R_av = self.get_delta_Q()

        q_l = ((modes_lav * delta_R_av).sum(axis=-1)
               * np.sqrt(self.masses_a)).sum(axis=-1)
        S_l = 1 / 2 * omega_l * q_l**2 * s**2

        return S_l, q_l

    def get_info(self):
        S_l, q_l = self.get_partial_hr()
        omega_l, modes_lav = self.get_eigenvalues_and_eigenvectors()
        dQ = np.sqrt((np.abs(q_l)**2).sum())
        W = (1 / dQ**2 * (np.abs(q_l)**2 * omega_l**2).sum())**0.5

        print(f'delta_Q: {dQ:.4}')
        print(f'Avg w: {W:.4}')

    def get_elph_function(self):

        omega_k = self.omega_k
        omega_l, modes_lav = self.get_eigenvalues_and_eigenvectors()

        S_l, q_l = self.get_partial_hr()

        S_k = 0

        for l in range(omega_l.shape[0]):
            S_k += S_l[l] * self.broadening(omega_k, omega_l[l], self.sigma)

        return S_k

    def get_spectral_function(self):

        gamma = self.gamma
        omega_k = self.omega_k
        dw = abs(omega_k[1] - omega_k[0])
        S_k = self.get_elph_function()
        S_l, q_l = self.get_partial_hr()
        S_0 = S_l.sum()

        times_i = np.fft.rfftfreq(omega_k.size, d=dw) * 2 * pi
        St_i = np.fft.rfft(S_k) * dw

        # Generating function
        Gt_n = np.exp(St_i - S_0) * np.exp(-gamma * times_i)

        # Spectral function
        Aw_k = np.fft.fftshift(np.fft.irfft(Gt_n))
        Aw_k = Aw_k / (np.trapz(Aw_k, dx=dw))

        return Aw_k

    def plot_elph_function(self, ax, filename=None):

        omega_l, modes_lav = self.get_eigenvalues_and_eigenvectors()
        S_l = self.get_partial_hr()
        S_k = self.get_elph_function()

        ax.plot(self.omega_k,
                S_k * np.max(S_l) / np.max(S_k), '-k')

        ax.set_xlabel(r'$\omega$ (meV)', size=15)
        ax.set_ylabel(r'S($\omega$)', size=15)
        ax.set_xlim(0, omega_l[-1] * 1.1)
        ax.set_ylim(0, np.max(S_l) * 1.1)
        ax.tick_params(labelsize=12)
        dict1 = {
            "omega(meV)": self.omega_k,
            "S(omega)": S_k * np.max(S_l) / np.max(S_k), }
        write_json(f'{filename}.json', dict1)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)

    def plot_spectral_function(self, ax, transition='emission',
                               ZPL=2.1, color='-C2',
                               label='Generating function',
                               filename=None):

        Aw_k = self.get_spectral_function()

        if transition == 'emission':
            ax.plot(-self.omega_k + np.max(self.omega_k) / 2 + ZPL,
                    Aw_k, color, lw=2., label=label)
        if transition == 'absorption':
            ax.plot(self.omega_k - np.max(self.omega_k) / 2 + ZPL,
                    Aw_k, color, lw=2., label=label)
        dict = {"Energy": - self.omega_k + np.max(self.omega_k) / 2 + ZPL,
                "PL Intensity": Aw_k, }

        write_json(f'{filename}.json', dict)
        ax.set_ylim(0, max(Aw_k) * 1.2)
        ax.set_xlim(0, ZPL * 1.3)
        ax.set_xlabel(r'$\omega$ (eV)', size=14)
        ax.set_ylabel(r'A($\omega$)', size=14)
        ax.tick_params(labelsize=12)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
