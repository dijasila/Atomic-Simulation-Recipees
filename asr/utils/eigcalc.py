from __future__ import annotations
from math import pi
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpaw.new.ase_interface import GPAW


class EigCalc:
    cell_cv: np.ndarray

    def get_band(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_new_band(self,
                     kind: str,
                     kpt_xv: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GPAWEigenvalueCalculator(EigCalc):
    def __init__(self,
                 calc: GPAW,
                 theta: float = None,  # degrees
                 phi: float = None):  # degrees
        """Do SOC calculation.

        Parameters
        ----------
        calc:
            GPAW ground-state object.
        theta:
            Polar angle in degrees.
        phi:
            Azimuthal angle in degrees.

        Returns
        -------
        * k-point vectors
        * eigenvalues
        * PAW-projections
        * spin-projections
        """
        self.calc = calc
        self.theta = theta
        self.phi = phi
        self.cell_cv = calc.atoms.cell

        if theta is None and phi is None:
            eig_Kn, fermilevel = self._eigs()
        else:
            assert theta is not None and phi is not None
            eig_Kn, fermilevel = self._soc_eigs()

        self.nocc = (eig_Kn[0] < fermilevel).sum()

        self.bands = {'vbm': eig_Kn[:, self.nocc - 1].copy(),
                      'cbm': eig_Kn[:, self.nocc].copy()}

        kpt_Kc = calc.wfs.kd.bzk_kc
        self.kpt_Kv = 2 * pi * kpt_Kc @ np.linalg.inv(self.cell_cv).T
        self.pbc_c = calc.wfs.kd.N_c > 1

    def get_band(self, kind):
        return self.kpt_Kv, self.bands[kind]

    def get_new_band(self, kind, kpt_xv):
        from gpaw.spinorbit import soc_eigenstates

        kpt_xc = kpt_xv @ self.cell_cv.T / (2 * pi)
        nsc_calc = self.calc.fixed_density(
            kpts=kpt_xc,
            symmetry='off',
            txt=f'{kind}.txt')
        if self.theta is None:
            nkpts = len(kpt_xv)
            nspins = self.calc.get_number_of_spins()
            eig_skn = np.array(
                [[nsc_calc.get_eigenvalues(kpt=kpt, spin=spin)
                  for kpt in range(nkpts)]
                 for spin in range(nspins)])
            eig_kns = eig_skn.transpose((1, 2, 0))
            eig_kn = eig_kns.reshape((nkpts, -1))
            eig_kn.sort(axis=1)
        else:
            states = soc_eigenstates(nsc_calc, theta=self.theta, phi=self.phi)
            eig_kn = states.eigenvalues()

        if kind == 'vbm':
            return eig_kn[:, self.nocc - 1].copy()
        else:
            return eig_kn[:, self.nocc].copy()

    def _eigs(self):
        kd = self.calc.wfs.kd
        nibzkpts = kd.nibzkpts
        nspins = self.calc.get_number_of_spins()
        eig_skn = np.array(
            [[self.calc.get_eigenvalues(kpt=kpt, spin=spin)
              for kpt in range(nibzkpts)]
             for spin in range(nspins)])
        eig_kns = eig_skn.transpose((1, 2, 0))
        eig_kn = eig_kns.reshape((nibzkpts, -1))
        eig_kn.sort(axis=1)
        eig_Kn = eig_kn[kd.bz2ibz_k]
        fermilevel = self.calc.get_fermi_level()
        return eig_Kn, fermilevel

    def _soc_eigs(self):
        # Extract SOC eigenvalues:
        from gpaw.spinorbit import soc_eigenstates

        states = soc_eigenstates(self.calc, theta=self.theta, phi=self.phi)
        kd = self.calc.wfs.kd
        kpt_Kc = np.array([kd.bzk_kc[wf.bz_index]
                           for wf in states])
        assert (kpt_Kc == kd.bzk_kc).all()
        eig_Kn = states.eigenvalues()
        fermilevel = states.fermi_level
        return eig_Kn, fermilevel
