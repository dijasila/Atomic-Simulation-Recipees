"""Effective masses - version 117."""
# noqa: W504
import pickle
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from ase.units import Bohr, Ha

from asr.magnetic_anisotropy import get_spin_axis
from asr.nppdpoly import PolyFit

if TYPE_CHECKING:
    from gpaw.calculator import GPAW
    from gpaw.kpt_descriptor import KPointDescriptor
else:
    GPAW = None
    KPointDescriptor = None


def extract_stuff_from_gpw_file(gpwpath: Path,
                                soc: False,
                                outpath: Path = None) -> None:
    from gpaw.calculator import GPAW
    calc = GPAW(gpwpath)
    stuff = extract_stuff_from_gpaw_calculation(calc, soc)
    outpath.write_bytes(pickle.dumps(stuff))
    return stuff


def extract_stuff_from_gpaw_calculation(calc: GPAW) -> Dict[str, Any]:
    from gpaw.spinorbit import soc_eigenstates
    assert calc.world.size == 1
    kd: KPointDescriptor = calc.wfs.kd
    theta, phi = get_spin_axis()
    states = soc_eigenstates(calc, theta=theta, phi=phi)
    k_kc = np.array([kd.bzk_kc[wf.bz_index]
                     for wf in states])
    eig_kn = np.array([wf.eig_m
                       for wf in states])
    proj_knI = np.array([wf.projections.matrix.array
                         for wf in states])
    spinproj_knv = states.spin_projections()
    fermilevel = states.fermi_level

    nocc = (eig_kn[0] < fermilevel).sum()
    N = range(nocc - 4, nocc + 4)
    K1, K2, K3 = tuple(kd.N_c)
    _, _, nI = proj_knI.shape
    return {'cell_cv': calc.atoms.cell,
            'kpt_ijkc': k_kc.reshape((K1, K2, K3, 3)),
            'fermilevel': fermilevel,
            'eig_ijkn': eig_kn.reshape((K1, K2, K3, -1))[..., N],
            'proj_ijknI': proj_knI.reshape(
                (K1, K2, K3, -1, nI))[..., N, :].astype(np.complex64),
            'spinproj_ijknv': spinproj_knv.reshape(
                (K1, K2, K3, -1, 3))[..., N, :].astype(np.float16)}


def connect(eig_ijkn, fingerprint_ijknx, threshold=2.0):
    K1, K2, K3, N = fingerprint_ijknx.shape[:-1]
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

    return eig_ijkn


def con2(e_kn, fingerprint_knx):
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


def main(data: dict,
         log=print):
    for kind in ['vbm', 'cbm']:
        k_ijkc, e_ijkn, axes, gap = find_extrema(
            kind=kind,
            **data)
        print(k_ijkc.shape, e_ijkn.shape, axes, gap)

        cell_cv = data['cell_cv'][axes][:, axes]

        k_kc = k_ijkc.reshape((-1, 3))[:, axes]
        e_kn = e_ijkn.reshape((-1, e_ijkn.shape[3]))

        extrema = []
        for e_k in e_kn.T:
            try:
                k_v, energy, mass_w, direction_wv, error_k = fit(
                    k_kc, e_k, None, cell_cv)
            except NoMinimum:
                pass
            else:
                if kind == 'vbm':
                    energy *= -1
                extrema.append((k_v, energy, mass_w, direction_wv, error_k))

        print(extrema)

        if kind == 'vbm':
            vbm = bm = max(extrema, key=lambda x: x[1])
        else:
            cbm = bm = min(extrema, key=lambda x: x[1])

        k_v, energy, mass_w, direction_wv, error_k = bm
        log(f'{kind}:')
        log('K-point:', k2str(k_v, cell_cv))
        log(f'Energy:  {energy:.3f} eV')
        log(f'Mass:    {mass_w} m_e')

    diff = cbm[1] - vbm[1] - gap
    assert -0.01 < diff <= 0.0, diff

    return vbm, cbm


def find_extrema(cell_cv,
                 kpt_ijkc,
                 fermilevel,
                 eig_ijkn,
                 proj_ijknI,
                 spinproj_ijknv=None,
                 kind='cbm',
                 log=print,
                 npoints=3):
    assert kind in ['vbm', 'cbm']

    nocc = (eig_ijkn[0, 0, 0] < fermilevel).sum()
    gap = eig_ijkn[..., nocc].min() - eig_ijkn[..., nocc - 1].max()
    print(eig_ijkn[..., nocc].min(), eig_ijkn[..., nocc - 1].max())
    log(f'Occupied bands: {nocc}')
    log(f'Fermi level: {fermilevel} eV')
    log(f'Gap: {gap} eV')
    log(proj_ijknI.shape)
    K1, K2, K3, N, _ = proj_ijknI.shape

    if spinproj_ijknv is None:
        spinproj_ijknv = np.zeros((K1, K2, K3, N, 3))

    if kind == 'cbm':
        # bands = slice(nocc, N)
        bands = slice(nocc, nocc + 6)
        eig_ijkn = eig_ijkn[..., bands]
    else:
        bands = slice(nocc - 1, nocc - 7 if nocc - 7 >= 0 else None, -1)
        # bands = slice(nocc - 1, nocc-4, -1)
        eig_ijkn = -eig_ijkn[..., bands]

    proj_ijknI = proj_ijknI[..., bands, :]
    spinproj_ijknv = spinproj_ijknv[..., bands, :]

    ijk = eig_ijkn[:, :, :, 0].ravel().argmin()
    i, j, k = np.unravel_index(ijk, (K1, K2, K3))

    dk = 3
    r1 = [0] if K1 == 1 else [x % K1 for x in range(i - dk, i + dk + 1)]
    r2 = [0] if K2 == 1 else [x % K2 for x in range(j - dk, j + dk + 1)]
    r3 = [0] if K3 == 1 else [x % K3 for x in range(k - dk, k + dk + 1)]
    e_ijkn = eig_ijkn[r1][:, r2][:, :, r3]
    fingerprint_ijknI = proj_ijknI[r1][:, r2][:, :, r3]
    k_ijkc = (kpt_ijkc[r1][:, r2][:, :, r3] - kpt_ijkc[i, j, k] + 0.5) % 1
    k_ijkc += kpt_ijkc[i, j, k] - 0.5

    log('Connecting bands')
    connect(e_ijkn, fingerprint_ijknI)

    axes = [c for c, size in enumerate([K1, K2, K3]) if size > 1]

    return k_ijkc, e_ijkn, axes, gap


class NoMinimum(ValueError):
    """Band doesn't have a minumum."""


def k2str(k_v, cell_cv):
    k_c = cell_cv @ k_v / (2 * pi)
    v = ', '.join(f'{k:7.3f}' for k in k_v)
    c = ', '.join(f'{k:6.3f}' for k in k_c)
    return f'({v}) Ang^-1 = ({c})'


def fit(k_kc, eig_k, spinproj_kv,
        cell_cv,
        npoints=None):
    dims = k_kc.shape[1]
    npoints = npoints or [7, 25, 55][dims - 1]

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


def cli():
    import sys
    path = Path(sys.argv[1])
    if path.suffix == '.gpw':
        stuff = extract_stuff_from_gpw_file(path, True,
                                            path.with_suffix('.pckl'))
    else:
        stuff = pickle.loads(path.read_bytes())
    main(stuff)


if __name__ == '__main__':
    cli()
