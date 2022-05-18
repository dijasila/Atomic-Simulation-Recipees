"""Effective masses - version 117."""
from __future__ import annotations
from math import pi

import numpy as np
from ase.units import Bohr, Ha

from asr.magnetic_anisotropy import get_spin_axis
from asr.utils.ndpoly import PolyFit
from asr.core import command, ASRResult


def extract_soc_stuff_from_gpaw_calculation(calc,
                                            theta: float = 0.0,
                                            phi: float = 0.0,
                                            ) -> tuple[np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray]:
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
    from gpaw.spinorbit import soc_eigenstates
    kd = calc.wfs.kd
    states = soc_eigenstates(calc, theta=theta, phi=phi)
    k_kc = np.array([kd.bzk_kc[wf.bz_index]
                     for wf in states])
    eig_kn = np.array([wf.eig_m
                       for wf in states])
    proj_knI = np.array([wf.projections.matrix.array
                         for wf in states])
    spinproj_knv = states.spin_projections()
    fermilevel = states.fermi_level
    eig_kn -= fermilevel

    K1, K2, K3 = tuple(kd.N_c)
    _, N, I = proj_knI.shape
    return (k_kc.reshape((K1, K2, K3, 3)),
            eig_kn.reshape((K1, K2, K3, N)),
            proj_knI.reshape((K1, K2, K3, N, I)),
            spinproj_knv.reshape((K1, K2, K3, N, 3)))


def connect(eig_ijkn: np.ndarray,
            fingerprint_ijknx: np.ndarray) -> None:
    """Reorder eigenvalues to give connected bands."""
    K1, K2, K3, N = eig_ijkn.shape
    for k1 in range(K1 - 1):
        con2(eig_ijkn[k1:k1 + 2, 0, 0],
             fingerprint_ijknx[k1:k1 + 2, 0, 0])

    for k1 in range(K1):
        for k2 in range(K2 - 1):
            con2(eig_ijkn[k1, k2:k2 + 2, 0],
                 fingerprint_ijknx[k1, k2:k2 + 2, 0])

    for k1 in range(K1):
        for k2 in range(K2):
            for k3 in range(K3 - 1):
                con2(eig_ijkn[k1, k2, k3:k3 + 2],
                     fingerprint_ijknx[k1, k2, k3:k3 + 2])


def con2(e_kn: np.ndarray,
         fingerprint_knx: np.ndarray) -> list[int]:
    """Connect 2 k-points."""
    K, N = e_kn.shape
    assert K == 2

    ovl_n1n2 = abs(fingerprint_knx[0] @ fingerprint_knx[1].conj().T)

    n2_n1 = []
    n1_n2: dict[int, int] = {}
    for n1 in range(N):
        n2 = ovl_n1n2[n1].argmax()
        ovl_n1n2[:, n2] = -1.0
        n2_n1.append(n2)
        n1_n2[n2] = n1

    fingerprint2_nx = fingerprint_knx[1].copy()

    e2_n = e_kn[1, n2_n1]
    fingerprint2_nx = fingerprint2_nx[n2_n1]
    e_kn[1] = e2_n
    fingerprint_knx[1] = fingerprint2_nx

    return n2_n1


class EMassesResult(ASRResult):
    vbm_k_v: list[float]
    vbm_mass_w: list[float]
    vbm_direction_wv: list[list[float]]
    cbm_k_v: list[float]
    cbm_mass_w: list[float]
    cbm_direction_wv: list[list[float]]

    key_descriptions = {
        'vbm_k_v': 'Position of VBM [Ang^-1]',
        'vbm_mass_w': 'VBM masses [m_e]',
        'vbm_dirction_wv': 'VBM directions',
        'cbm_k_v': 'Position of CBM [Ang^-1]',
        'cbm_mass_w': 'CBM masses [m_e]',
        'cbm_dirction_wv': 'CBM directions'}


@command('asr.emasses2',
         requires=['pdos.gpw'],
         dependencies=['asr.pdos'])
def main() -> EMassesResult:
    """"""
    from gpaw import GPAW
    calc = GPAW('pdos.gpw')

    theta, phi = get_spin_axis()

    K_ijkc, eig_ijkn, proj_ijknI, spinproj_ijknv = \
        extract_soc_stuff_from_gpaw_calculation(calc, theta, phi)
    cell_cv = calc.atoms.cell

    extrema = _main(cell_cv, K_ijkc, eig_ijkn, proj_ijknI)
    dct = {}
    for xbm, (k_v, energy, mass_w, dir_wv) in zip(['vbm', 'cbm'], extrema):
        dct[f'{xbm}_k_v'] = k_v
        dct[f'{xbm}_mass_w'] = mass_w
        dct[f'{xbm}_direction_wv'] = dir_wv

    return EMassesResult.fromdata(**dct)


class BadFitError(ValueError):
    """Bad fit to data."""


def _main(cell_cv: np.ndarray,
          K_ijkc: np.ndarray,
          eig_ijkn: np.ndarray,
          proj_ijknI: np.ndarray) -> list[tuple[np.ndarray,
                                                float,
                                                np.ndarray,
                                                np.ndarray]]:
    nocc = (eig_ijkn[0, 0, 0] < 0.0).sum()

    K1, K2, K3 = K_ijkc.shape[:3]
    axes = [c for c, size in enumerate([K1, K2, K3]) if size > 1]
    cell_cv = cell_cv[axes][:, axes]

    extrema = []
    for kind in ['vbm', 'cbm']:
        if kind == 'cbm':
            E_ijkn = eig_ijkn[..., nocc:nocc + 4]
            P_ijknI = proj_ijknI[..., nocc:nocc + 4, :]
        else:
            E_ijkn = -eig_ijkn[..., max(nocc - 4, 0):nocc][..., ::-1]
            P_ijknI = proj_ijknI[..., max(nocc - 4, 0):nocc, :][..., ::-1, :]

        k_kc, e_kn = find_minima(K_ijkc, E_ijkn, P_ijknI)

        k_kc = k_kc[:, axes]

        for e_k in e_kn.T:
            try:
                k_v, energy, mass_w, direction_wv, error_k = fit(
                    k_kc, e_k, cell_cv)
            except NoMinimum:
                pass
            else:
                break
        else:  # no break
            raise NoMinimum

        error = abs(error_k).max()
        if error > max(0.02 * e_k.ptp(), 0.001):
            raise BadFitError(f'Error: {error} eV')

        error = abs(energy - e_kn.min())
        if error > 0.1:
            raise BadFitError(f'Error: {error} eV')

        if kind == 'vbm':
            energy *= -1

        extrema.append((k_v, energy, mass_w, direction_wv))

    return extrema


def find_minima(kpt_ijkc: np.ndarray,
                eig_ijkn: np.ndarray,
                proj_ijknI: np.ndarray,
                spinproj_ijknv: np.ndarray = None,
                npoints: int = 3) -> tuple[np.ndarray, np.ndarray]:
    K1, K2, K3, N, _ = proj_ijknI.shape

    if spinproj_ijknv is None:
        spinproj_ijknv = np.zeros((K1, K2, K3, N, 3))

    ijk = eig_ijkn[:, :, :, 0].ravel().argmin()
    i, j, k = np.unravel_index(ijk, (K1, K2, K3))

    dk = 3
    r1 = [0] if K1 == 1 else [x % K1 for x in range(i - dk, i + dk + 1)]
    r2 = [0] if K2 == 1 else [x % K2 for x in range(j - dk, j + dk + 1)]
    r3 = [0] if K3 == 1 else [x % K3 for x in range(k - dk, k + dk + 1)]

    e_ijkn = eig_ijkn[r1][:, r2][:, :, r3].copy()
    fingerprint_ijknI = proj_ijknI[r1][:, r2][:, :, r3].copy()
    k_ijkc = (kpt_ijkc[r1][:, r2][:, :, r3] - kpt_ijkc[i, j, k] + 0.5) % 1
    k_ijkc += kpt_ijkc[i, j, k] - 0.5

    connect(e_ijkn, fingerprint_ijknI)

    e_ijkn.sort()

    return k_ijkc.reshape((-1, 3)), e_ijkn.reshape((-1, N))


class NoMinimum(ValueError):
    """Band doesn't have a minimum."""


def fit(k_kc: np.ndarray,
        eig_k: np.ndarray,
        cell_cv: np.ndarray) -> tuple[np.ndarray,
                                      float,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray]:
    dims = k_kc.shape[1]
    npoints = [7, 25, 55][dims - 1]

    k0_c = k_kc[eig_k.argmin()]
    if (k0_c <= k_kc.min(0)).any() or (k0_c >= k_kc.max(0)).any():
        raise NoMinimum('Minimum too close to edge of box!')

    k_kv = k_kc @ np.linalg.inv(cell_cv).T * 2 * pi
    k0_v = k_kv[eig_k.argmin()].copy()

    k_kv -= k0_v
    k = (k_kv**2).sum(1).argsort()[:npoints]

    if len(k) < npoints:
        raise NoMinimum('Too few points!')

    k_kv = k_kv[k]
    eig_k = eig_k[k]

    try:
        # fit = Fit3D(k_kv, eig_k)
        fit = PolyFit(k_kv, eig_k, 4)
    except np.linalg.LinAlgError:
        raise NoMinimum('Bad minimum!')

    hessian_vv = fit.hessian(np.zeros(dims))
    eval_w = np.linalg.eigvalsh(hessian_vv)
    if eval_w.min() <= 0.0:
        raise NoMinimum('Not a minimum')

    error_k = np.array([fit.value(k_v) - e for k_v, e in zip(k_kv, eig_k)])

    k_v = fit.find_minimum()
    emin = fit.value(k_v)
    hessian_vv = fit.hessian(k_v)
    evals_w, evec_vw = np.linalg.eigh(hessian_vv)
    mass_w = Bohr**2 * Ha / evals_w

    if (mass_w < 0.01).any():
        raise NoMinimum('Unrealistic mass!')

    return k_v + k0_v, emin, mass_w, evec_vw.T, error_k


if __name__ == '__main__':
    main.cli()
