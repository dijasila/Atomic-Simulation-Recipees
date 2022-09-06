"""Effective masses - version 117-b."""
from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from ase.units import Bohr, Ha

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  fig, make_panel_description)
from asr.magnetic_anisotropy import get_spin_axis
# from asr.utils.ndpoly import PolyFit, nterms

if TYPE_CHECKING:
    from gpaw.new.ase_interface import GPAW


class MassDict(TypedDict):
    kind: str
    k_v: list[float]
    mass_w: list[float]
    direction_wv: list[list[float]]
    energy: float
    max_fit_error: float
    fit_data_i: list[float]


class EigCalc:
    cell_cv: np.ndarray

    def get_band(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_new_band(self,
                     kind: str,
                     kpt_xv: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@command('asr.emasses2',
         requires=['gs.gpw'],
         dependencies=['asr.gs'])
def main() -> ASRResult:
    """Find effective masses."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw')

    theta, phi = get_spin_axis()
    eigcalc = GPAWEigenvalueCalculator(calc, theta, phi)
    vbm = _main(eigcalc, 'vbm')
    cbm = _main(eigcalc, 'cbm')
    return EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)


def _main(eigcalc: EigCalc,
          kind: str,
          maxlevels: int = 9) -> MassDict:
    """Find effective masses."""
    kpt_ijkv, eig_ijk = eigcalc.get_band(kind)
    shape = eig_ijk.shape
    if kind == 'vbm':
        i, j, k = np.unravel_index(eig_ijk.ravel().argmax(), shape)
    else:
        i, j, k = np.unravel_index(eig_ijk.ravel().argmin(), shape)
    kpt0_v = kpt_ijkv[i, j, k].copy()

    N = 3
    I, J, K = [[0] if n == 1 else
               [x % n for x in range(x0 - N, x0 + N + 1)]
               for n, x0 in zip(shape, [i, j, k])]
    print(I, J, K)
    print(i, j, k, kind, eig_ijk[i, j, k])
    eig_ijk = eig_ijk[I][:, J][:, :, K]
    kpt_ijkv = kpt_ijkv[I][:, J][:, :, K]
    kpt_ijkc = kpt_ijkv @ eigcalc.cell_cv.T / (2 * pi)
    kpt0_c = kpt0_v @ eigcalc.cell_cv.T / (2 * pi)
    kpt_ijkc += 0.5 - kpt0_c
    kpt_ijkc %= 1
    kpt_ijkc -= 0.5 - kpt0_c
    kpt_ijkv = 2 * pi * kpt_ijkc @ np.linalg.inv(eigcalc.cell_cv).T
    massdata = find_mass(kpt_ijkv, eig_ijk, kind, eigcalc, maxlevels)
    return massdata


class BadFitError(ValueError):
    """Bad fit to data."""


def find_mass(kpt_ijkv: np.ndarray,
              eig_ijk: np.ndarray,
              kind: str = 'cbm',
              eigcalc=None,
              maxlevels: int = 1) -> MassDict:
    if kind == 'vbm':
        eig_ijk = -eig_ijk

    shape = eig_ijk.shape
    i, j, k = (int(x) for x in
               np.unravel_index(eig_ijk.ravel().argmin(), shape))
    print('K')
    print(kpt_ijkv.min((0, 1, 2)))
    print(kpt_ijkv.max((0, 1, 2)))
    #mask = basin(eig_ijk, i, j, k).ravel()
    eig_x = eig_ijk.ravel()#[mask]
    kpt_xv = kpt_ijkv.reshape((-1, 3))#[mask]
    print(eig_x.shape)
    axes = [c for c, size in enumerate(eig_ijk.shape) if size > 1]
    kpt_xv = kpt_xv[:, axes]
    print(axes)

    try:
        kpt_v, energy, mass_w, direction_wv, error_x, fit = fit_band(
            kpt_xv, eig_x)
    except BadFitError as xx:
        print(xx)
    else:
        if kind == 'vbm':
            energy *= -1
            fit.coefs *= -1
        fit_error = abs(error_x).max()
        return MassDict(kind=kind,
                        k_v=kpt_v.tolist(),
                        energy=energy,
                        mass_w=mass_w.tolist(),
                        direction_wv=direction_wv.tolist(),
                        max_fit_error=fit_error,
                        fit_data_i=fit.coefs.tolist())

    if maxlevels == 1:
        raise ValueError

    kpt_xv = kpt_ijkv.reshape((-1, 3))
    knndist = ((kpt_xv[1:] - kpt_xv[0])**2).sum(1).min()**0.5
    kspan = 2.5 * knndist
    kspan = 1*knndist

    # Create box of k-points centered on kpt0_v:
    kpt0_v = kpt_ijkv[i, j, k]
    kpt_vx = [np.linspace(kpt - kspan / 2, kpt + kspan / 2, npoints)
              if npoints > 1
              else np.array([0.0])
              for kpt, npoints in zip(kpt0_v, shape)]
    print('KSPAN', kspan, kpt0_v)
    kpt_ijkv = np.empty(shape + (3,))
    kpt_ijkv[..., 0] = kpt_vx[0][:, np.newaxis, np.newaxis]
    kpt_ijkv[..., 1] = kpt_vx[1][:, np.newaxis]
    kpt_ijkv[..., 2] = kpt_vx[2]

    eig_x = eigcalc.get_new_band(kind, kpt_ijkv.reshape((-1, 3)))
    eig_ijk = eig_x.reshape(shape)
    return find_mass(kpt_ijkv, eig_ijk, kind, eigcalc, maxlevels - 1)


def fit_band(k_xv: np.ndarray,
             eig_x: np.ndarray,
             order: int = 4,
             max_rel_error: float = 0.02) -> tuple[np.ndarray,
                                                   float,
                                                   np.ndarray,
                                                   np.ndarray,
                                                   np.ndarray,
                                                   PolyFit]:
    npoints, ndims = k_xv.shape
    nterms = (order // 2 + 1) * (order + 1) + 4
    if npoints < 1.25 * nterms:
        raise BadFitError('Too few points!')

    kmin_v = k_xv[eig_x.argmin()].copy()
    dk_xv = k_xv - kmin_v
    f = YFit(dk_xv, eig_x, order)
    from scipy.optimize import minimize
    r = minimize(f, [0, 0, 0], method='Nelder-Mead')
    print(r)
    """
    error_x = np.array([fit.value(k_v) - e for k_v, e in zip(dk_xv, eig_x)])
    fit_error = abs(error_x).max()
    hessian_vv = fit.hessian(np.zeros_like(kmin_v))
    print('Error', fit_error)
    eval_w = np.linalg.eigvalsh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise BadFitError('Not a minimum')

    dkmin_v = fit.find_minimum()
    emin = fit.value(dkmin_v)
    hessian_vv = fit.hessian(dkmin_v)
    eval_w, evec_vw = np.linalg.eigh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise BadFitError('Not a minimum')
    mass_w = Bohr**2 * Ha / eval_w
    print(dkmin_v + kmin_v, emin, mass_w)

    if fit_error > max_rel_error * eig_x.ptp():
        raise BadFitError(f'Error too big: {fit_error} eV', eig_x.ptp(),
                          fit_error / eig_x.ptp())

    # Centered fit around minumum:
    fit = PolyFit(dk_xv - dkmin_v, eig_x, order=4)

    return dkmin_v + kmin_v, emin, mass_w, evec_vw.T, error_x, fit
    """


def basin(eig_ijk: np.ndarray,
          i: int,
          j: int,
          k: int) -> np.ndarray:
    """...

    >>> eigs = np.array([[[1, 0, 1, 0.5]]])
    >>> basin(eigs, 0, 0, 1)
    """
    include = {(i, j, k): True}
    mask_ijk = np.empty(eig_ijk.shape, bool)
    for i0 in range(eig_ijk.shape[0]):
        for j0 in range(eig_ijk.shape[1]):
            for k0 in range(eig_ijk.shape[2]):
                ok = _basin(eig_ijk, i0, j0, k0, include)
                mask_ijk[i0, j0, k0] = ok
    return mask_ijk


def _basin(eig_ijk: np.ndarray,
           i: int,
           j: int,
           k: int,
           include: dict[tuple[int, int, int], bool]) -> bool:
    ijk = (i, j, k)
    if ijk in include:
        return include[ijk]

    candidates = [(eig_ijk[i, j, k], ijk)]
    x_c = list(ijk)
    for d in [-1, 1]:
        for c in [0, 1, 2]:
            x_c[c] += d
            if 0 <= x_c[c] < eig_ijk.shape[c]:
                candidates.append((eig_ijk[x_c[0], x_c[1], x_c[2]],
                                   (x_c[0], x_c[1], x_c[2])))
            x_c[c] -= d
    _, ijk0 = min(candidates)
    if ijk0 == ijk:
        ok = False
    else:
        ok = _basin(eig_ijk, *ijk0, include)
    include[ijk] = ok
    return ok


panel_description = make_panel_description(
    """The effective mass tensor represents the second derivative of the band
energy w.r.t. wave vector at a band extremum. The effective masses of the
valence bands (VB) and conduction bands (CB) are obtained as the eigenvalues
of the mass tensor.  Spin–orbit interactions are included.""")


def webpanel(result, row, key_descriptions):
    rows = []
    for data in [result.cbm_mass, result.vbm_mass]:
        for n, mass in enumerate(data['mass_w'], start=1):
            kind = data['kind']
            rows.append([f'{kind}, direction {n}',
                         f'{mass:.2f} m<sub>0</sub>'])
    table = {'type': 'table',
             'header': ['', 'value'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.emasses2')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Effective masses',
                                     description=title_description),
             'columns': [[fig('cbm-mass.png'), table],
                         [fig('vbm-mass.png')]],
             'plot_descriptions':
                 [{'function': mass_plots,
                   'filenames': ['cbm-mass.png',
                                 'vbm-mass.png']}]}

    return [panel]


@prepare_result
class EMassesResult(ASRResult):
    vbm_mass: MassDict
    cbm_mass: MassDict

    key_descriptions = {
        'vbm_mass': 'Mass data for VBM',
        'cbm_mass': 'Mass data for CBM'}

    formats = {'ase_webpanel': webpanel}


def mass_plots(row, *filenames):
    result = row.data.get('results-asr.emasses2.json')
    cbm = result.cbm_mass
    vbm = result.vbm_mass

    x_p = np.linspace(-0.2, 0.2, 51)
    y_bwp = []
    height = 0.0
    offset = -vbm['energy']
    for data in [cbm, vbm]:
        y_wp = []
        dir_wv = np.array(data['direction_wv'])
        fit = PolyFit.from_coefs(coefs=data['fit_data_i'],
                                 order=4,
                                 ndims=len(dir_wv))
        for dir_v in dir_wv:
            y_p = [fit.value(x * dir_v) + offset for x in x_p]
            height = max(height, np.ptp(y_p))
            y_wp.append(y_p)
        y_bwp.append(y_wp)

    height *= 1.1
    plots = []
    for data, y_wp, filename in zip([cbm, vbm], y_bwp, filenames):
        dir_wv = data['direction_wv']
        fig, ax = plt.subplots()
        for n, (dir_v, y_p, ls) in enumerate(zip(dir_wv,
                                                 y_wp,
                                                 ['-', '--', ':']),
                                             1):
            d = ','.join(f'{d:.2f}' for d in dir_v)
            ax.plot(x_p, y_p,
                    ls=ls,
                    label=f'direction {n}: [{d}]')
        plots.append(ax)
        ax.legend()
        y0 = (np.max(y_wp) + np.min(y_wp)) / 2
        ax.axis(ymin=y0 - height / 2, ymax=y0 + height / 2)
        ax.set_xlabel(r'$\Delta k [1/Å]$')
        ax.set_ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
        fig.tight_layout()
        fig.savefig(filename)
    return plots


class GPAWEigenvalueCalculator(EigCalc):
    def __init__(self,
                 calc: GPAW,
                 theta: float = None,  # degrees
                 phi: float = None):  # degrees
        """Do SOC calculation.

        Parameters
        ----------
        calc:
            GPAW ground-state object
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

        shape = tuple(calc.wfs.kd.N_c)
        eig_ijkn = eig_Kn.reshape(shape + (-1,))
        self.bands = {'vbm': eig_ijkn[..., self.nocc - 1].copy(),
                      'cbm': eig_ijkn[..., self.nocc].copy()}

        kpt_ijkc = calc.wfs.kd.bzk_kc.reshape(shape + (3,))
        self.kpt_ijkv = 2 * pi * kpt_ijkc @ np.linalg.inv(self.cell_cv).T

    def get_band(self, kind):
        return self.kpt_ijkv, self.bands[kind]

    def get_new_band(self, kind, kpt_xv):
        from gpaw.spinorbit import soc_eigenstates

        kpt_xc = kpt_xv @ self.cell_cv.T / (2 * pi)
        nsc_calc = self.calc.fixed_density(
            kpts=kpt_xc,
            symmetry='off',
            convergence={'bands': 0 if kind == 'vbm' else -1},
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


if __name__ == '__main__':
    main.cli()


class YFit:
    def __init__(self,
                 k_iv: np.ndarray,
                 eig_i: np.ndarray,
                 lmax: int = 2):
        self.k_iv = k_iv
        self.eig_i = eig_i
        assert lmax % 2 == 0
        self.lmax = lmax

    def fit(self, k_v):
        k_iv = self.k_iv - k_v
        k2_i = (k_iv**2).sum(1)
        eps = 1e-12
        k2_i[k2_i < eps] = eps
        khat_iv = k_iv / (k2_i**0.5)[:, np.newaxis]
        J = (self.lmax // 2 + 1) * (self.lmax + 1) + 1
        M_ji = np.empty((J, len(k2_i)))
        M_ji[0] = 1.0
        j = 1
        for l in range(0, self.lmax + 1, 2):
            for m in range(2 * l + 1):
                x_i = Y(l, m, *khat_iv.T)
                M_ji[j] = x_i * k2_i
                j += 1
        c_j = np.linalg.lstsq(M_ji.T, self.eig_i, rcond=None)[0]
        return c_j, M_ji

    def __call__(self, k_v):
        c_j, M_ji = self.fit(k_v)
        return ((c_j @ M_ji - self.eig_i)**2).sum()


def Y(l, m, x, y, z):
    result = 0.0
    for c, (i, j, k) in YL[l**2 + m]:
        result += c * x**i * y**j * z**k
    return result


YL = [
    # s:
    [(1, (0, 0, 0))],
    # p:
    [(1, (0, 1, 0))],
    [(1, (0, 0, 1))],
    [(1, (1, 0, 0))],
    # d:
    [(1, (1, 1, 0))],
    [(1, (0, 1, 1))],
    [(2, (0, 0, 2)), (-1, (0, 2, 0)), (-1, (2, 0, 0))],
    [(1, (1, 0, 1))],
    [(1, (2, 0, 0)), (-1, (0, 2, 0))],
    # f:
    [(-1, (0, 3, 0)), (3, (2, 1, 0))],
    [(1, (1, 1, 1))],
    [(-1, (0, 3, 0)), (4, (0, 1, 2)), (-1, (2, 1, 0))],
    [(2, (0, 0, 3)), (-3, (2, 0, 1)), (-3, (0, 2, 1))],
    [(4, (1, 0, 2)), (-1, (3, 0, 0)), (-1, (1, 2, 0))],
    [(1, (2, 0, 1)), (-1, (0, 2, 1))],
    [(1, (3, 0, 0)), (-3, (1, 2, 0))],
    # g:
    [(1, (3, 1, 0)), (-1, (1, 3, 0))],
    [(-1, (0, 3, 1)), (3, (2, 1, 1))],
    [(-1, (3, 1, 0)), (-1, (1, 3, 0)), (6, (1, 1, 2))],
    [(-3, (2, 1, 1)), (4, (0, 1, 3)), (-3, (0, 3, 1))],
    [(6, (2, 2, 0)), (-24, (2, 0, 2)), (3, (0, 4, 0)),
     (-24, (0, 2, 2)), (3, (4, 0, 0)), (8, (0, 0, 4))],
    [(4, (1, 0, 3)), (-3, (3, 0, 1)), (-3, (1, 2, 1))],
    [(6, (2, 0, 2)), (1, (0, 4, 0)), (-1, (4, 0, 0)), (-6, (0, 2, 2))],
    [(1, (3, 0, 1)), (-3, (1, 2, 1))],
    [(-6, (2, 2, 0)), (1, (0, 4, 0)), (1, (4, 0, 0))],
    # h:
    [(-10, (2, 3, 0)), (5, (4, 1, 0)), (1, (0, 5, 0))],
    [(1, (3, 1, 1)), (-1, (1, 3, 1))],
    [(-8, (0, 3, 2)), (1, (0, 5, 0)), (-3, (4, 1, 0)), (-2, (2, 3, 0)),
     (24, (2, 1, 2))],
    [(-1, (3, 1, 1)), (-1, (1, 3, 1)), (2, (1, 1, 3))],
    [(-12, (0, 3, 2)), (2, (2, 3, 0)), (-12, (2, 1, 2)), (8, (0, 1, 4)),
     (1, (4, 1, 0)), (1, (0, 5, 0))],
    [(30, (2, 2, 1)), (-40, (0, 2, 3)), (15, (0, 4, 1)), (-40, (2, 0, 3)),
     (15, (4, 0, 1)), (8, (0, 0, 5))],
    [(-12, (3, 0, 2)), (8, (1, 0, 4)), (1, (5, 0, 0)), (2, (3, 2, 0)),
     (-12, (1, 2, 2)), (1, (1, 4, 0))],
    [(-1, (4, 0, 1)), (1, (0, 4, 1)), (2, (2, 0, 3)), (-2, (0, 2, 3))],
    [(8, (3, 0, 2)), (-1, (5, 0, 0)), (2, (3, 2, 0)), (-24, (1, 2, 2)),
     (3, (1, 4, 0))],
    [(1, (4, 0, 1)), (-6, (2, 2, 1)), (1, (0, 4, 1))],
    [(-10, (3, 2, 0)), (1, (5, 0, 0)), (5, (1, 4, 0))],
    # i:
    [(3, (5, 1, 0)), (-10, (3, 3, 0)), (3, (1, 5, 0))],
    [(5, (4, 1, 1)), (-10, (2, 3, 1)), (1, (0, 5, 1))],
    [(10, (3, 1, 2)), (-1, (5, 1, 0)), (1, (1, 5, 0)), (-10, (1, 3, 2))],
    [(-8, (0, 3, 3)), (-6, (2, 3, 1)), (3, (0, 5, 1)), (24, (2, 1, 3)),
     (-9, (4, 1, 1))],
    [(-16, (3, 1, 2)), (16, (1, 1, 4)), (2, (3, 3, 0)), (1, (5, 1, 0)),
     (-16, (1, 3, 2)), (1, (1, 5, 0))],
    [(5, (0, 5, 1)), (-20, (0, 3, 3)), (10, (2, 3, 1)), (-20, (2, 1, 3)),
     (5, (4, 1, 1)), (8, (0, 1, 5))],
    [(90, (4, 0, 2)), (-120, (0, 2, 4)), (-15, (2, 4, 0)), (16, (0, 0, 6)),
     (-15, (4, 2, 0)), (90, (0, 4, 2)), (-5, (0, 6, 0)), (-120, (2, 0, 4)),
     (-5, (6, 0, 0)), (180, (2, 2, 2))],
    [(-20, (3, 0, 3)), (8, (1, 0, 5)), (5, (5, 0, 1)), (10, (3, 2, 1)),
     (-20, (1, 2, 3)), (5, (1, 4, 1))],
    [(16, (2, 0, 4)), (-16, (0, 2, 4)), (-1, (2, 4, 0)), (-16, (4, 0, 2)),
     (1, (4, 2, 0)), (-1, (0, 6, 0)), (1, (6, 0, 0)), (16, (0, 4, 2))],
    [(8, (3, 0, 3)), (-3, (5, 0, 1)), (6, (3, 2, 1)), (-24, (1, 2, 3)),
     (9, (1, 4, 1))],
    [(5, (4, 2, 0)), (10, (0, 4, 2)), (-60, (2, 2, 2)), (-1, (0, 6, 0)),
     (-1, (6, 0, 0)), (10, (4, 0, 2)), (5, (2, 4, 0))],
    [(1, (5, 0, 1)), (-10, (3, 2, 1)), (5, (1, 4, 1))],
    [(-15, (4, 2, 0)), (-1, (0, 6, 0)), (1, (6, 0, 0)), (15, (2, 4, 0))]]
