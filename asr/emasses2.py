"""Effective masses - version 117."""
# noqa: W504
import pickle
from itertools import combinations_with_replacement
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
    return stuff


def extract_stuff_from_gpaw_calculation(calc: GPAW,
                                        soc: False) -> Dict[str, Any]:
    assert calc.world.size == 1
    kd: KPointDescriptor = calc.wfs.kd
    if soc:
        from gpaw.spinorbit import soc_eigenstates
        states = soc_eigenstates(calc)  # , scale=1)
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


def con2(e_kn, fp_knx, verbose=False):
    K, N = e_kn.shape
    assert K == 2, K

    ovl_n1n2 = abs(fp_knx[0] @ fp_knx[1].conj().T)

    c2 = []  # clusters(e_kn[1])

    for a, b in c2:
        o = ovl_n1n2[:, a:b]
        o[:] = o.max(axis=1)[:, np.newaxis]

    if verbose:
        print(ovl_n1n2)

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

    if verbose:
        print(n2_n1, n1_n2)

    e2_n = e_kn[1, n2_n1]
    fp2_nx = fp2_nx[n2_n1]
    e_kn[1] = e2_n
    fp_knx[1] = fp2_nx

    return n2_n1


def main(data: dict,
         log=print):
    for kind in ['vbm', 'cbm']:
        k_ijkc, e_ijkn, axes, gap = find_extrema(
            kind=kind,
            log=lambda *args: None,
            **data)

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
    fp_ijknI = proj_ijknI[r1][:, r2][:, :, r3]
    k_ijkc = (kpt_ijkc[r1][:, r2][:, :, r3] - kpt_ijkc[i, j, k] + 0.5) % 1
    k_ijkc += kpt_ijkc[i, j, k] - 0.5

    log('Connecting bands')
    connect(e_ijkn, fp_ijknI)

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


def deriv(f, d):
    return [f[:i] + f[i + 1:] for i, x in enumerate(f) if x == d]


def tuples2str(tuples):
    if not tuples:
        return '0'
    assert len(set(tuples)) == 1
    return '*'.join([str(len(tuples))] + ['xyz'[d] for d in tuples[0]])


class PolyFit:
    def __init__(self, x, y, order=2, verbose=False):
        ndims = x.shape[1]
        self.ndims = ndims

        t0 = []
        for n in range(order + 1):
            t0.extend(combinations_with_replacement(range(ndims), n))
        args = ', '.join('xyz'[:ndims])
        s0 = ', '.join(tuples2str([t]) for t in t0)
        self.f0 = eval(compile(f'lambda {args}: [{s0}]', '', 'eval'))

        t1 = [[deriv(t, d) for t in t0] for d in range(ndims)]
        s1 = '], ['.join(', '.join(tuples2str(tt)
                                   for tt in t1[d])
                         for d in range(ndims))
        self.f1 = eval(compile(f'lambda {args}: [[{s1}]]', '', 'eval'))

        t2 = [[[sum((deriv(t, d2) for t in tt), start=[]) for tt in t1[d1]]
               for d1 in range(ndims)]
              for d2 in range(ndims)]
        s2 = ']], [['.join('], ['.join(', '.join(tuples2str(tt)
                                                 for tt in t2[d1][d2])
                                       for d1 in range(ndims))
                           for d2 in range(ndims))
        self.f2 = eval(compile(f'lambda {args}: [[[{s2}]]]', '', 'eval'))

        M = self.f0(*x.T)
        M[0] = np.ones(len(x))
        M = np.array(M)
        self.coefs = np.linalg.solve(M @ M.T, M @ y)

        if verbose:
            print(f'[{s0}]')
            print(f'[[{s1}]]')
            print(f'[[[{s2}]]]')
            print(self.coefs)

    def value(self, k_v):
        return self.f0(*k_v) @ self.coefs

    def gradient(self, k_v):
        return self.f1(*k_v) @ self.coefs

    def hessian(self, k_v):
        return self.f2(*k_v) @ self.coefs

    def find_minimum(self, k_v=None):
        from scipy.optimize import minimize

        def f(k_v):
            return self.value(k_v), self.gradient(k_v)

        if k_v is None:
            k_v = np.zeros(self.ndims)

        result = minimize(f, k_v, jac=True, method='Newton-CG')
        return result.x


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
