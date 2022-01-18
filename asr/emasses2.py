"""Effective masses - version 117."""
# noqa: W504
import pickle
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from ase.units import Bohr, Ha

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


def extract_stuff_from_gpaw_calculation(calc: GPAW,
                                        soc: False) -> Dict[str, Any]:
    assert calc.world.size == 1
    kd: KPointDescriptor = calc.wfs.kd
    if soc:
        from gpaw.spinorbit import soc_eigenstates
        states = soc_eigenstates(calc)
        k_kc = np.array([kd.bzk_kc[wf.bz_index]
                         for wf in states])
        eig_kn = np.array([wf.eig_m
                           for wf in states])
        proj_knI = np.array([wf.projections.matrix.array
                             for wf in states])
        spinproj_knv = states.spin_projections()
        fermilevel = states.fermi_level
    else:
        k_kc = kd.bzk_kc
        assert kd.ibzk_kc.shape == k_kc.shape
        eig_ksn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                             for s in range(kd.nspins)]
                            for k in range(kd.nibzkpts)])
        proj_ksnI = np.array([[calc.wfs.kpt_qs[k][s].projections.collect()
                               for s in range(kd.nspins)]
                              for k in range(kd.nibzkpts)])
        K, S, N, nI = proj_ksnI.shape
        if kd.nspins == 2:
            proj1_ksnI = proj_ksnI
            proj_ksnI = np.zeros((K, S, N, 2 * nI))
            proj_ksnI[:, 0, :, :nI] = proj1_ksnI[:, 0]
            proj_ksnI[:, 1, :, nI:] = proj1_ksnI[:, 1]
            nI *= 2

        spinproj_ksnv = np.zeros((K, S, N, 3))
        for s in range(kd.nspins):
            spinproj_ksnv[:, s, :, 2] = 1 - 2 * s

        eig_kn = eig_ksn.reshape((K, S * N))
        proj_knI = proj_ksnI.reshape((K, S * N, nI))
        spinproj_knv = spinproj_ksnv.reshape((K, S * N, 3))
        n_kn = eig_kn.argsort(axis=1)
        eig_kn = np.take_along_axis(eig_kn, n_kn, axis=1)
        proj_knI = np.take_along_axis(proj_knI, n_kn[:, :, None], axis=1)
        spinproj_knv = np.take_along_axis(spinproj_knv, n_kn[:, :, None],
                                          axis=1)
        fermilevel = calc.get_fermi_level()

    nocc = (eig_kn[0, 0, 0] < fermilevel).sum()
    N = range(nocc - 4, nocc + 4)

    K1, K2, K3 = tuple(kd.N_c)
    _, N, nI = proj_knI.shape
    return {'cell_cv': calc.atoms.cell,
            'kpt_ijkc': k_kc.reshape((K1, K2, K3, 3)),
            'fermilevel': fermilevel,
            'eig_ijkn': eig_kn.reshape((K1, K2, K3, N)),
            'proj_ijknI': proj_knI.reshape((K1, K2, K3, N, nI)).astype(
                np.complex64),
            'spinproj_ijknv': spinproj_knv.reshape((K1, K2, K3, N, 3)).astype(
                np.float16)}


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


def clusters(eigs,
             eps: float = 1e-3) -> List[Tuple[int, int]]:
    """Find clusters of degenerate eigenvalues.

    >>> list(clusters(np.zeros(4)))
    [(0, 4)]
    >>> list(clusters(np.arange(4)))
    []
    >>> list(clusters(np.array([0, 0, 1, 1, 1, 2])))
    [(0, 2), (2, 5)]
    """
    e1 = eigs[0]
    n = 0
    c = []
    for i2, e2 in enumerate(eigs[1:], 1):
        if e2 - e1 < eps:
            n += 1
        else:
            e1 = e2
            if n:
                c.append((i2 - n - 1, i2))
                n = 0
    if n:
        c.append((i2 - n, i2 + 1))
    return c


def con2(e_kn, fp_knx):
    K, N = e_kn.shape
    assert K == 2, K

    ovl_n1n2 = abs(fp_knx[0] @ fp_knx[1].conj().T)

    c2 = clusters(e_kn[1])

    for a, b in c2:
        o = ovl_n1n2[:, a:b]
        o[:] = o.max(axis=1)[:, np.newaxis]

    n2_n1 = []
    n1_n2: dict[int, int] = {}
    for n1 in range(N):
        n2 = ovl_n1n2[n1].argmax()
        ovl_n1n2[:, n2] = -1.0
        n2_n1.append(n2)
        n1_n2[n2] = n1

    fp2_nx = fp_knx[1].copy()
    for a, b in c2:
        for n2 in range(a, b):
            fp2_nx[n2] = fp_knx[0, n1_n2[n2]]

    e2_n = e_kn[1, n2_n1]
    fp2_nx = fp2_nx[n2_n1]
    e_kn[1] = e2_n
    fp_knx[1] = fp2_nx

    return n2_n1


def main(data: dict,
         kind='cbm',
         log=print):
    k_ijkc, e_ijkn, axes = find_extrema(kind=kind,
                                        log=log,
                                        **data)

    cell_cv = data['cell_cv'][axes][:, axes]

    k_kv = k_ijkc.reshape((-1, 3))[:, axes] @ np.linalg.inv(cell_cv).T * 2 * pi
    e_kn = e_ijkn.reshape((-1, e_ijkn.shape[3]))

    for e_k in e_kn.T:
        k_v, energy, mass_w, direction_wv, error_k = fit(
            k_kv, e_k, None, cell_cv,
            log=log)


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
    log(f'Occupied bands: {nocc}')
    log(f'Fermi level: {fermilevel} eV')
    log(proj_ijknI.shape)
    K1, K2, K3, N, _ = proj_ijknI.shape

    if spinproj_ijknv is None:
        spinproj_ijknv = np.zeros((K1, K2, K3, N, 3))

    if kind == 'cbm':
        # bands = slice(nocc, N)
        bands = slice(nocc, nocc + 6)
        eig_ijkn = eig_ijkn[..., bands]
    else:
        bands = slice(nocc - 1, None, -1)
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
    fp_ijknI = proj_ijknI[r1][:, r2][:, :, r3]
    k_ijkc = (kpt_ijkc[r1][:, r2][:, :, r3] - kpt_ijkc[i, j, k] + 0.5) % 1
    k_ijkc += kpt_ijkc[i, j, k] - 0.5

    log('Connecting bands')
    connect(e_ijkn, fp_ijknI)

    axes = [c for c, size in enumerate([K1, K2, K3]) if size > 1]

    return k_ijkc, e_ijkn, axes


class NoMinimum(ValueError):
    """Band doesn't have a minumum."""


def k2str(k_v, cell_cv):
    k_c = cell_cv @ k_v / (2 * pi)
    v = ', '.join(f'{k:7.3f}' for k in k_v)
    c = ', '.join(f'{k:6.3f}' for k in k_c)
    return f'({v}) Ang^-1 = ({c})'


def fit(k_kv, eig_k, spinproj_kv,
        cell_cv,
        npoints=None,
        log=print):
    dims = k_kv.shape[1]
    npoints = npoints or [7, 15, 25][dims - 1]

    def K(k_v):
        return k2str(k_v, cell_cv)

    kmin_v = k_kv[eig_k.argmin()].copy()
    k_kv -= kmin_v
    k = (k_kv**2).sum(1).argsort()[:npoints]
    log(f'Fitting to {len(k)} points close to {K(kmin_v)}:')

    if len(k) < npoints:
        log('  Too few points!')
        raise NoMinimum

    k_kv = k_kv[k]
    eig_k = eig_k[k]

    try:
        fit = Fit3D(k_kv, eig_k)
    except np.linalg.LinAlgError:
        log('   Bad minimum!')
        raise NoMinimum

    hessian_vv = fit.hessian(np.zeros(dims))
    eval_w = np.linalg.eigvalsh(hessian_vv)
    if eval_w.min() <= 0.0:
        log('  Not a minimum')
        raise NoMinimum

    error_k = np.array([fit.value(k_v) - e for k_v, e in zip(k_kv, eig_k)])
    log(f'  Maximum error: {abs(error_k).max() * 1000:.3f} meV')

    k_v = fit.find_minimum()
    emin = fit.value(k_v)
    hessian_vv = fit.hessian(k_v)
    evals_w, evec_vw = np.linalg.eigh(hessian_vv)
    mass_w = Bohr**2 * Ha / evals_w

    log(f'  Found minimum: {K(k_v + kmin_v)}, {emin:.3f} eV')
    for w, (mass, evec_v) in enumerate(zip(mass_w, evec_vw.T)):
        log(f'    Mass #{w}: {mass:.3f} m_e, {K(evec_v)}')

    if (mass_w < 0.01).any():
        log('  Unrealistic mass!')
        raise NoMinimum

    return k_v + kmin_v, emin, mass_w, evec_vw.T, error_k


class Fit3D:
    def __init__(self, k_iv, eig_i):
        self.dims = k_iv.shape[1]
        if self.dims == 1:
            x = k_iv[:, 0]
            f_ji = np.array([x**0, x, x**2, x**3])
        elif self.dims == 2:
            x, y = k_iv.T
            f_ji = np.array(
                [x**0, x, y, x**2, y**2, x * y, x**3, y**3, x**2 * y, y**2 * x])
        else:
            x, y, z = k_iv.T
            f_ji = np.array(
                [x**0, x, y, z, x**2, y**2, z**2, x * y, y * z, z * x,
                 x**3, y**3, z**3,
                 x**2 * y, x**2 * z, y**2 * x, y**2 * z, z**2 * x, z**2 * y,
                 x * y * z])
        self.coef_j = np.linalg.solve(f_ji @ f_ji.T, f_ji @ eig_i)

    def find_minimum(self, k_v=None):
        from scipy.optimize import minimize

        def f(k_v):
            return self.value(k_v), self.gradient(k_v)

        if k_v is None:
            k_v = np.zeros(self.dims)

        result = minimize(f, k_v, jac=True, method='Newton-CG')
        return result.x

    def value(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            c, cx, cxx, cxxx = self.coef_j
            return c + cx * x + cxx * x**2 + cxxx * x**3
        if self.dims == 2:
            x, y = k_v
            c, cx, cy, cxx, cyy, cxy, cxxx, cyyy, cxxy, cxyy = self.coef_j
            return (
                c + cx * x + cy * y + cxx * x**2 + cxy * x * y +  # noqa: W504
                cyy * y**2 + cxxx * x**3 + cxxy * x**2 * y +  # noqa: W504
                cxyy * x * y**2 + cyyy * y**3)
        1 / 0

    def gradient(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            _, cx, cxx, cxxx = self.coef_j
            return [cx + 2 * cxx * x + 3 * cxxx * x**2]
        if self.dims == 2:
            x, y = k_v
            _, cx, cy, cxx, cyy, cxy, cxxx, cyyy, cxxy, cxyy = self.coef_j
            return [
                cx + 2 * cxx * x + cxy * y +  # noqa: W504
                3 * cxxx * x**2 + 2 * cxxy * x * y + cxyy * y**2,
                cy + 2 * cyy * y + cxy * x +  # noqa: W504
                3 * cyyy * y**2 + 2 * cxyy * x * y + cxxy * x**2]
        1 / 0

    def hessian(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            _, _, cxx, cxxx = self.coef_j
            return [[2 * cxx + 6 * cxxx * x]]
        if self.dims == 2:
            x, y = k_v
            _, _, _, cxx, cyy, cxy, cxxx, cyyy, cxxy, cxyy = self.coef_j
            return [[2 * cxx + 6 * cxxx * x + 2 * cxxy * y,
                     cxy + 2 * cxyy * y + 2 * cxxy * x],
                    [cxy + 2 * cxyy * y + 2 * cxxy * x,
                     2 * cyy + 6 * cyyy * y + 2 * cxyy * x]]
        1 / 0


def cli():
    import sys
    path = Path(sys.argv[1])
    if path.suffix == '.gpw':
        soc = bool(sys.argv[2])
        extract_stuff_from_gpw_file(path, soc, path.with_suffix('.pckl'))
    else:
        kind = sys.argv[2]
        data = pickle.loads(path.read_bytes())
        # k_v, energy, mass_w, direction_wv, error_k = ...
        things = main(data, kind)
        path.with_suffix(f'.{kind}.pckl').write_bytes(pickle.dumps(things))


if __name__ == '__main__':
    cli()
