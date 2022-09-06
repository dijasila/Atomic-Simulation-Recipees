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
        kpt_v, energy, mass_w, direction_wv, error_x, coefs = fit_band(
            kpt_xv, eig_x)
    except BadFitError as xx:
        print(xx)
    else:
        if kind == 'vbm':
            energy *= -1
            coefs *= -1
        fit_error = abs(error_x).max()
        return MassDict(kind=kind,
                        k_v=kpt_v.tolist(),
                        energy=energy,
                        mass_w=mass_w.tolist(),
                        direction_wv=direction_wv.tolist(),
                        max_fit_error=fit_error,
                        fit_data_i=coefs.tolist())

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
                                                   np.ndarray]:
    npoints, ndims = k_xv.shape
    nterms = (order // 2 + 1) * (order + 1) + 4
    if npoints < 1.25 * nterms:
        raise BadFitError('Too few points!')

    kmin_v = k_xv[eig_x.argmin()].copy()
    dk_xv = k_xv - kmin_v
    f = YFit(dk_xv, eig_x, order)
    from scipy.optimize import minimize
    result = minimize(
        f,
        x0=[0, 0, 0],
        method='Nelder-Mead')  # seems more robust than the default
    assert result.success
    dkmin_v = result.x
    coefs, error_x = f.fit(dkmin_v)
    fit_error = abs(error_x).max()
    print('Error', fit_error)
    hessian_vv = hessian(coefs)
    eval_w, evec_vw = np.linalg.eigh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise BadFitError('Not a minimum')

    emin = coefs[0]
    mass_w = Bohr**2 * Ha / eval_w
    print(dkmin_v + kmin_v, emin, mass_w)

    if fit_error > max_rel_error * eig_x.ptp():
        raise BadFitError(f'Error too big: {fit_error} eV', eig_x.ptp(),
                          fit_error / eig_x.ptp())
    return dkmin_v + kmin_v, emin, mass_w, evec_vw.T, error_x, coefs


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
        error_i = c_j @ M_ji - self.eig_i
        return c_j, error_i

    def __call__(self, k_v):
        _, error_i = self.fit(k_v)
        return (error_i**2).sum()


def hessian(coef_j):
    c00, c20, c21, c22, c23, c24 = coef_j[1:7]
    hess_vv = np.eye(3) * c00 * 2 * 0.28209479177387814
    hess_vv[0, 1] = c20 * 1.0925484305920792
    hess_vv[1, 2] = c21 * 1.0925484305920792
    hess_vv[2, 2] += c22 * 2 * 0.6307831305050401
    hess_vv[1, 1] -= c22 * 2 * 0.31539156525252005
    hess_vv[0, 0] -= c22 * 2 * 0.31539156525252005
    hess_vv[0, 2] = c23 * 1.0925484305920792
    hess_vv[0, 0] += c24 * 2 * 0.5462742152960396
    hess_vv[1, 1] -= c24 * 2 * 0.5462742152960396
    hess_vv[1, 0] = hess_vv[0, 1]
    hess_vv[2, 0] = hess_vv[0, 1]
    hess_vv[2, 1] = hess_vv[1, 2]
    return hess_vv


def Y(l, m, x, y, z):
    result = 0.0
    for c, (i, j, k) in Y_L[l**2 + m]:
        result += c * x**i * y**j * z**k
    return result


Y_L = [
    # s, l=0:
    [(0.28209479177387814, (0, 0, 0))],
    # p, l=1:
    [(0.4886025119029199, (0, 1, 0))],
    [(0.4886025119029199, (0, 0, 1))],
    [(0.4886025119029199, (1, 0, 0))],
    # d, l=2:
    [(1.0925484305920792, (1, 1, 0))],
    [(1.0925484305920792, (0, 1, 1))],
    [(0.6307831305050401, (0, 0, 2)),
     (-0.31539156525252005, (0, 2, 0)),
     (-0.31539156525252005, (2, 0, 0))],
    [(1.0925484305920792, (1, 0, 1))],
    [(0.5462742152960396, (2, 0, 0)),
     (-0.5462742152960396, (0, 2, 0))],
    # f, l=3:
    [(-0.5900435899266435, (0, 3, 0)),
     (1.7701307697799304, (2, 1, 0))],
    [(2.890611442640554, (1, 1, 1))],
    [(-0.4570457994644658, (0, 3, 0)),
     (1.828183197857863, (0, 1, 2)),
     (-0.4570457994644658, (2, 1, 0))],
    [(0.7463526651802308, (0, 0, 3)),
     (-1.1195289977703462, (2, 0, 1)),
     (-1.1195289977703462, (0, 2, 1))],
    [(1.828183197857863, (1, 0, 2)),
     (-0.4570457994644658, (3, 0, 0)),
     (-0.4570457994644658, (1, 2, 0))],
    [(1.445305721320277, (2, 0, 1)),
     (-1.445305721320277, (0, 2, 1))],
    [(0.5900435899266435, (3, 0, 0)),
     (-1.7701307697799304, (1, 2, 0))],
    # g, l=4:
    [(2.5033429417967046, (3, 1, 0)),
     (-2.5033429417967046, (1, 3, 0))],
    [(-1.7701307697799307, (0, 3, 1)),
     (5.310392309339792, (2, 1, 1))],
    [(-0.9461746957575601, (3, 1, 0)),
     (-0.9461746957575601, (1, 3, 0)),
     (5.6770481745453605, (1, 1, 2))],
    [(-2.0071396306718676, (2, 1, 1)),
     (2.676186174229157, (0, 1, 3)),
     (-2.0071396306718676, (0, 3, 1))],
    [(0.6347132814912259, (2, 2, 0)),
     (-2.5388531259649034, (2, 0, 2)),
     (0.31735664074561293, (0, 4, 0)),
     (-2.5388531259649034, (0, 2, 2)),
     (0.31735664074561293, (4, 0, 0)),
     (0.8462843753216345, (0, 0, 4))],
    [(2.676186174229157, (1, 0, 3)),
     (-2.0071396306718676, (3, 0, 1)),
     (-2.0071396306718676, (1, 2, 1))],
    [(2.8385240872726802, (2, 0, 2)),
     (0.47308734787878004, (0, 4, 0)),
     (-0.47308734787878004, (4, 0, 0)),
     (-2.8385240872726802, (0, 2, 2))],
    [(1.7701307697799307, (3, 0, 1)),
     (-5.310392309339792, (1, 2, 1))],
    [(-3.755014412695057, (2, 2, 0)),
     (0.6258357354491761, (0, 4, 0)),
     (0.6258357354491761, (4, 0, 0))],
    # h, l=5:
    [(-6.5638205684017015, (2, 3, 0)),
     (3.2819102842008507, (4, 1, 0)),
     (0.6563820568401701, (0, 5, 0))],
    [(8.302649259524165, (3, 1, 1)),
     (-8.302649259524165, (1, 3, 1))],
    [(-3.913906395482003, (0, 3, 2)),
     (0.4892382994352504, (0, 5, 0)),
     (-1.467714898305751, (4, 1, 0)),
     (-0.9784765988705008, (2, 3, 0)),
     (11.741719186446009, (2, 1, 2))],
    [(-4.793536784973324, (3, 1, 1)),
     (-4.793536784973324, (1, 3, 1)),
     (9.587073569946648, (1, 1, 3))],
    [(-5.435359814348363, (0, 3, 2)),
     (0.9058933023913939, (2, 3, 0)),
     (-5.435359814348363, (2, 1, 2)),
     (3.6235732095655755, (0, 1, 4)),
     (0.45294665119569694, (4, 1, 0)),
     (0.45294665119569694, (0, 5, 0))],
    [(3.508509673602708, (2, 2, 1)),
     (-4.678012898136944, (0, 2, 3)),
     (1.754254836801354, (0, 4, 1)),
     (-4.678012898136944, (2, 0, 3)),
     (1.754254836801354, (4, 0, 1)),
     (0.9356025796273888, (0, 0, 5))],
    [(-5.435359814348363, (3, 0, 2)),
     (3.6235732095655755, (1, 0, 4)),
     (0.45294665119569694, (5, 0, 0)),
     (0.9058933023913939, (3, 2, 0)),
     (-5.435359814348363, (1, 2, 2)),
     (0.45294665119569694, (1, 4, 0))],
    [(-2.396768392486662, (4, 0, 1)),
     (2.396768392486662, (0, 4, 1)),
     (4.793536784973324, (2, 0, 3)),
     (-4.793536784973324, (0, 2, 3))],
    [(3.913906395482003, (3, 0, 2)),
     (-0.4892382994352504, (5, 0, 0)),
     (0.9784765988705008, (3, 2, 0)),
     (-11.741719186446009, (1, 2, 2)),
     (1.467714898305751, (1, 4, 0))],
    [(2.075662314881041, (4, 0, 1)),
     (-12.453973889286246, (2, 2, 1)),
     (2.075662314881041, (0, 4, 1))],
    [(-6.5638205684017015, (3, 2, 0)),
     (0.6563820568401701, (5, 0, 0)),
     (3.2819102842008507, (1, 4, 0))],
    # i, l=6:
    [(4.099104631151485, (5, 1, 0)),
     (-13.663682103838287, (3, 3, 0)),
     (4.099104631151485, (1, 5, 0))],
    [(11.83309581115876, (4, 1, 1)),
     (-23.66619162231752, (2, 3, 1)),
     (2.366619162231752, (0, 5, 1))],
    [(20.182596029148968, (3, 1, 2)),
     (-2.0182596029148967, (5, 1, 0)),
     (2.0182596029148967, (1, 5, 0)),
     (-20.182596029148968, (1, 3, 2))],
    [(-7.369642076119388, (0, 3, 3)),
     (-5.527231557089541, (2, 3, 1)),
     (2.7636157785447706, (0, 5, 1)),
     (22.108926228358165, (2, 1, 3)),
     (-8.29084733563431, (4, 1, 1))],
    [(-14.739284152238776, (3, 1, 2)),
     (14.739284152238776, (1, 1, 4)),
     (1.842410519029847, (3, 3, 0)),
     (0.9212052595149235, (5, 1, 0)),
     (-14.739284152238776, (1, 3, 2)),
     (0.9212052595149235, (1, 5, 0))],
    [(2.9131068125936572, (0, 5, 1)),
     (-11.652427250374629, (0, 3, 3)),
     (5.8262136251873144, (2, 3, 1)),
     (-11.652427250374629, (2, 1, 3)),
     (2.9131068125936572, (4, 1, 1)),
     (4.660970900149851, (0, 1, 5))],
    [(5.721228204086558, (4, 0, 2)),
     (-7.628304272115411, (0, 2, 4)),
     (-0.9535380340144264, (2, 4, 0)),
     (1.0171072362820548, (0, 0, 6)),
     (-0.9535380340144264, (4, 2, 0)),
     (5.721228204086558, (0, 4, 2)),
     (-0.3178460113381421, (0, 6, 0)),
     (-7.628304272115411, (2, 0, 4)),
     (-0.3178460113381421, (6, 0, 0)),
     (11.442456408173117, (2, 2, 2))],
    [(-11.652427250374629, (3, 0, 3)),
     (4.660970900149851, (1, 0, 5)),
     (2.9131068125936572, (5, 0, 1)),
     (5.8262136251873144, (3, 2, 1)),
     (-11.652427250374629, (1, 2, 3)),
     (2.9131068125936572, (1, 4, 1))],
    [(7.369642076119388, (2, 0, 4)),
     (-7.369642076119388, (0, 2, 4)),
     (-0.46060262975746175, (2, 4, 0)),
     (-7.369642076119388, (4, 0, 2)),
     (0.46060262975746175, (4, 2, 0)),
     (-0.46060262975746175, (0, 6, 0)),
     (0.46060262975746175, (6, 0, 0)),
     (7.369642076119388, (0, 4, 2))],
    [(7.369642076119388, (3, 0, 3)),
     (-2.7636157785447706, (5, 0, 1)),
     (5.527231557089541, (3, 2, 1)),
     (-22.108926228358165, (1, 2, 3)),
     (8.29084733563431, (1, 4, 1))],
    [(2.522824503643621, (4, 2, 0)),
     (5.045649007287242, (0, 4, 2)),
     (-30.273894043723452, (2, 2, 2)),
     (-0.5045649007287242, (0, 6, 0)),
     (-0.5045649007287242, (6, 0, 0)),
     (5.045649007287242, (4, 0, 2)),
     (2.522824503643621, (2, 4, 0))],
    [(2.366619162231752, (5, 0, 1)),
     (-23.66619162231752, (3, 2, 1)),
     (11.83309581115876, (1, 4, 1))],
    [(-10.247761577878714, (4, 2, 0)),
     (-0.6831841051919143, (0, 6, 0)),
     (0.6831841051919143, (6, 0, 0)),
     (10.247761577878714, (2, 4, 0))],
    # j, l=7:
    [(14.850417383016522, (2, 5, 0)),
     (4.950139127672174, (6, 1, 0)),
     (-24.75069563836087, (4, 3, 0)),
     (-0.7071627325245963, (0, 7, 0))],
    [(-52.91921323603801, (3, 3, 1)),
     (15.875763970811402, (1, 5, 1)),
     (15.875763970811402, (5, 1, 1))],
    [(-2.5945778936013015, (6, 1, 0)),
     (2.5945778936013015, (4, 3, 0)),
     (-62.26986944643124, (2, 3, 2)),
     (4.670240208482342, (2, 5, 0)),
     (6.226986944643123, (0, 5, 2)),
     (31.13493472321562, (4, 1, 2)),
     (-0.5189155787202603, (0, 7, 0))],
    [(41.513246297620825, (3, 1, 3)),
     (12.453973889286246, (1, 5, 1)),
     (-41.513246297620825, (1, 3, 3)),
     (-12.453973889286246, (5, 1, 1))],
    [(-18.775072063475285, (2, 3, 2)),
     (-0.4693768015868821, (0, 7, 0)),
     (0.4693768015868821, (2, 5, 0)),
     (2.3468840079344107, (4, 3, 0)),
     (-12.516714708983523, (0, 3, 4)),
     (37.55014412695057, (2, 1, 4)),
     (1.4081304047606462, (6, 1, 0)),
     (9.387536031737643, (0, 5, 2)),
     (-28.162608095212928, (4, 1, 2))],
    [(13.27598077334948, (3, 3, 1)),
     (6.63799038667474, (1, 5, 1)),
     (-35.402615395598616, (3, 1, 3)),
     (21.24156923735917, (1, 1, 5)),
     (-35.402615395598616, (1, 3, 3)),
     (6.63799038667474, (5, 1, 1))],
    [(-0.4516580379125865, (0, 7, 0)),
     (10.839792909902076, (0, 5, 2)),
     (-1.3549741137377596, (2, 5, 0)),
     (-1.3549741137377596, (4, 3, 0)),
     (-21.679585819804153, (0, 3, 4)),
     (-21.679585819804153, (2, 1, 4)),
     (5.781222885281108, (0, 1, 6)),
     (-0.4516580379125865, (6, 1, 0)),
     (21.679585819804153, (2, 3, 2)),
     (10.839792909902076, (4, 1, 2))],
    [(-11.471758521216831, (2, 0, 5)),
     (1.0925484305920792, (0, 0, 7)),
     (-11.471758521216831, (0, 2, 5)),
     (28.67939630304208, (2, 2, 3)),
     (-2.3899496919201733, (6, 0, 1)),
     (-7.16984907576052, (4, 2, 1)),
     (14.33969815152104, (4, 0, 3)),
     (-2.3899496919201733, (0, 6, 1)),
     (-7.16984907576052, (2, 4, 1)),
     (14.33969815152104, (0, 4, 3))],
    [(10.839792909902076, (1, 4, 2)),
     (-0.4516580379125865, (7, 0, 0)),
     (21.679585819804153, (3, 2, 2)),
     (-1.3549741137377596, (5, 2, 0)),
     (-0.4516580379125865, (1, 6, 0)),
     (-21.679585819804153, (3, 0, 4)),
     (-1.3549741137377596, (3, 4, 0)),
     (5.781222885281108, (1, 0, 6)),
     (-21.679585819804153, (1, 2, 4)),
     (10.839792909902076, (5, 0, 2))],
    [(10.620784618679584, (2, 0, 5)),
     (-10.620784618679584, (0, 2, 5)),
     (3.31899519333737, (6, 0, 1)),
     (3.31899519333737, (4, 2, 1)),
     (-17.701307697799308, (4, 0, 3)),
     (-3.31899519333737, (0, 6, 1)),
     (-3.31899519333737, (2, 4, 1)),
     (17.701307697799308, (0, 4, 3))],
    [(-1.4081304047606462, (1, 6, 0)),
     (0.4693768015868821, (7, 0, 0)),
     (18.775072063475285, (3, 2, 2)),
     (-0.4693768015868821, (5, 2, 0)),
     (12.516714708983523, (3, 0, 4)),
     (-2.3468840079344107, (3, 4, 0)),
     (28.162608095212928, (1, 4, 2)),
     (-37.55014412695057, (1, 2, 4)),
     (-9.387536031737643, (5, 0, 2))],
    [(10.378311574405206, (4, 0, 3)),
     (-3.1134934723215615, (0, 6, 1)),
     (15.56746736160781, (2, 4, 1)),
     (-62.26986944643124, (2, 2, 3)),
     (10.378311574405206, (0, 4, 3)),
     (-3.1134934723215615, (6, 0, 1)),
     (15.56746736160781, (4, 2, 1))],
    [(-2.5945778936013015, (1, 6, 0)),
     (-62.26986944643124, (3, 2, 2)),
     (-0.5189155787202603, (7, 0, 0)),
     (31.13493472321562, (1, 4, 2)),
     (2.5945778936013015, (3, 4, 0)),
     (6.226986944643123, (5, 0, 2)),
     (4.670240208482342, (5, 2, 0))],
    [(2.6459606618019005, (6, 0, 1)),
     (-2.6459606618019005, (0, 6, 1)),
     (-39.68940992702851, (4, 2, 1)),
     (39.68940992702851, (2, 4, 1))],
    [(0.7071627325245963, (7, 0, 0)),
     (-14.850417383016522, (5, 2, 0)),
     (24.75069563836087, (3, 4, 0)),
     (-4.950139127672174, (1, 6, 0))]]


if __name__ == '__main__':
    main.cli()
