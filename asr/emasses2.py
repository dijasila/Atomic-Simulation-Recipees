"""Effective masses."""
import pickle
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ase.units import Bohr, Ha
from gpaw.calculator import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.spinorbit import soc_eigenstates


def extract_stuff_from_gpw_file(gpwpath: Path,
                                soc: False,
                                outpath: Path = None) -> None:
    calc = GPAW(gpwpath)
    stuff = extract_stuff_from_gpaw_calculation(calc)
    outpath.write_bytes(pickle.dumps(stuff))


def extract_stuff_from_gpaw_calculation(calc: GPAW,
                                        soc: False) -> Dict[str, Any]:
    assert calc.world.size == 1
    kd: KPointDescriptor = calc.wfs.kd
    if soc:
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
        K, S, N, I = proj_ksnI.shape
        if kd.nspins == 2:
            proj1_ksnI = proj_ksnI
            proj_ksnI = np.zeros((K, S, N, 2 * I))
            proj_ksnI[:, 0, :, :I] = proj1_ksnI[:, 0]
            proj_ksnI[:, 1, :, I:] = proj1_ksnI[:, 1]
            I *= 2

        spinproj_ksnv = np.zeros((K, S, N, 3))
        for s in range(kd.nspins):
            spinproj_ksnv[:, s, :, 2] = 1 - 2 * s

        eig_kn = eig_ksn.reshape((K, S * N))
        proj_knI = proj_ksnI.reshape((K, S * N, I))
        spinproj_knv = spinproj_ksnv.reshape((K, S * N, 3))
        n_kn = eig_kn.argsort(axis=1)
        eig_kn = np.take_along_axis(eig_kn, n_kn, axis=1)
        proj_knI = np.take_along_axis(proj_knI, n_kn[:, :, None], axis=1)
        spinproj_knv = np.take_along_axis(spinproj_knv, n_kn[:, :, None],
                                          axis=1)
        fermilevel = calc.get_fermi_level()

    K1, K2, K3 = tuple(kd.N_c)
    _, N, I = proj_knI.shape
    return {'cell_cv': calc.atoms.cell,
            'kpt_ijkc': k_kc.reshape((K1, K2, K3, 3)),
            'fermilevel': fermilevel,
            'eig_ijkn': eig_kn.reshape((K1, K2, K3, N)),
            'proj_ijknI': proj_knI.reshape((K1, K2, K3, N, I)),
            'spinproj_ijknv': spinproj_knv.reshape((K1, K2, K3, N, 3))}


def connect(fingerprint_ijknx, threshold=2.0):
    K1, K2, K3, N = fingerprint_ijknx.shape[:-1]
    band_ijkn = np.zeros((K1, K2, K3, N), int) - 1
    bnew = 0
    equal = []
    for k1 in range(K1):
        for k2 in range(K2):
            bnew = con1d(fingerprint_ijknx[k1, k2], band_ijkn[k1, k2],
                         bnew, equal)
    for k1 in range(K1):
        for k3 in range(K3):
            bnew = con1d(fingerprint_ijknx[k1, :, k3], band_ijkn[k1, :, k3],
                         bnew, equal)
    for k2 in range(K2):
        for k3 in range(K3):
            bnew = con1d(fingerprint_ijknx[:, k2, k3], band_ijkn[:, k2, k3],
                         bnew, equal)

    mapping = {}
    for i, j in equal:
        i, j = sorted([i, j])
        if i not in mapping:
            mapping[j] = i
        else:
            mapping[j] = mapping[i]

    band_ijkn = np.array([mapping.get(b, b)
                          for b in band_ijkn.ravel()]).reshape(band_ijkn.shape)

    count = defaultdict(int)
    for b in band_ijkn.ravel():
        count[b] += 1

    mapping = dict(zip(sorted(count, key=lambda k: -count[k]),
                       range(len(count))))

    band_ijkn = np.array([mapping[b]
                          for b in band_ijkn.ravel()]).reshape(band_ijkn.shape)

    return band_ijkn


def con1d(fp_knx,
          b_kn,
          bnew,
          equal):
    K, N = fp_knx.shape[:2]
    for k in range(K - 1):
        ovl_n1n2 = abs(fp_knx[k] @ fp_knx[k + 1].conj().T)
        taken = set()
        for n2 in range(N):
            n1b, n1a = ovl_n1n2[:, n2].argsort()[-2:]
            b2 = b_kn[k + 1, n2]
            if ovl_n1n2[n1a, n2] > 2 * ovl_n1n2[n1b, n2] and n1a not in taken:
                b1 = b_kn[k, n1a]
                if b1 == -1:
                    b1 = bnew
                    bnew += 1
                    b_kn[k, n1a] = b1
                taken.add(n1a)
                if b2 == -1:
                    b_kn[k + 1, n2] = b1
                else:
                    if b1 != b2:
                        equal.append((b1, b2))
            else:
                if b2 == -1:
                    b_kn[k + 1, n2] = bnew
                    bnew += 1
    return bnew


def main(datapath: Path,
         kind='cbm',
         nbands=4):
    dct = pickle.loads(datapath.read_bytes())
    extrema = find_extrema(kind, nbands, **dct)
    return extrema


def find_extrema(cell_cv,
                 kpt_ijkc,
                 fermilevel,
                 eig_ijkn,
                 proj_ijknI,
                 spinproj_ijknv=None,
                 kind='cbm',
                 nbands=4,
                 log=print,
                 npoints=3):
    nocc = (eig_ijkn[0, 0, 0] < fermilevel).sum()
    log(nocc)

    K1, K2, K3, N, _ = proj_ijknI.shape

    if spinproj_ijknv is None:
        spinproj_ijknv = np.zeros((K1, K2, K3, N, 3))

    if kind == 'cbm':
        bands = slice(nocc, nocc + nbands)
        eig_ijkn = eig_ijkn[..., bands] - fermilevel
    else:
        bands = slice(nocc - 1, nocc - 1 - nbands, -1)
        eig_ijkn = fermilevel - eig_ijkn[..., bands]

    proj_ijknI = proj_ijknI[..., bands, :]
    spinproj_ijknv = spinproj_ijknv[..., bands, :]

    ijk = eig_ijkn[..., 0].ravel().argmin()
    i, j, k = np.unravel_index(ijk, eig_ijkn.shape[:3])
    log(i, j, k)
    log(kpt_ijkc[i, j, k])
    log(eig_ijkn[i, j, k])

    a, b, c = (0 if size == 1 else npoints for size in kpt_ijkc.shape[:3])
    A = np.arange(i - a, i + a + 1)
    B = np.arange(j - b, j + b + 1)
    C = np.arange(k - c, k + c + 1)

    eig_ijkn = eig_ijkn[A][:, B][:, :, C]
    spinproj_ijknv = spinproj_ijknv[A][:, B][:, :, C]
    print(spinproj_ijknv.shape)
    b_ijkn = connect(proj_ijknI[A][:, B][:, :, C])

    k_ijkc = np.indices(
        (2 * a + 1, 2 * b + 1, 2 * c + 1)).transpose((1, 2, 3, 0))
    k_ijkc += (i - a, j - b, k - c)
    log(k_ijkc.shape)
    k_ijkv = k_ijkc @ np.linalg.inv(cell_cv).T * 2 * pi

    axes = [c for c, size in enumerate([K1, K2, K3]) if size > 1]
    bands = []
    for b in range(nbands):
        mask_ijkn = b_ijkn == b
        eig_k = eig_ijkn[mask_ijkn]
        print(spinproj_ijknv.shape, mask_ijkn.shape)
        spinproj_kv = np.array([spinproj_ijknv[..., v][mask_ijkn]
                                for v in range(3)]).T
        k_kv = k_ijkv[mask_ijkn.any(axis=3)][:, axes]
        bands.append((k_kv, eig_k, spinproj_kv))

    return bands


def fit(k_kv, eig_k, spinproj_kv, npoints=None):
    dims = k_kv.shape[1]
    npoints = npoints or [7, 15, 25][dims - 1]

    kmin_v = k_kv[eig_k.argmin()]
    k_kv -= kmin_v
    k = (k_kv**2).sum(1).argsort()[:npoints]
    print(k, kmin_v)
    k_kv = k_kv[k]
    eig_k = eig_k[k]
    fit = Fit3D(k_kv, eig_k)
    hessian_vv = fit.hessian(np.zeros(dims))
    evals = np.linalg.eigvalsh(hessian_vv)
    if evals.min() <= 0.0:
        raise ValueError('Not a minimum')
    o = fit.find_minimum()
    print(o)
    return o
    # mass= 0.5 * Bohr**2 * Ha / poly[0]


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

        o = minimize(f, k_v, jac=True)
        return o

    def value(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            c, cx, cxx, cxxx = self.coef_j
            return c + cx * x + cxx * x**2 + cxxx * x**3
        if self.dims == 2:
            x, y = k_v
            c, cx, cy, cxx, cxy, cyy, cxxx, cxxy, cxyy, cyyy = self.coef_j
            return (c +
                    cx * x + cy * y +
                    cxx * x**2 + cxy * x * y + cyy * y**2 +
                    cxxx * x**3 + cxxy * x**2 * y +
                    cxyy * x * y**2 + cyyy * y**3)
        1 / 0

    def gradient(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            _, cx, cxx, cxxx = self.coef_j
            return [cx + 2 * cxx * x + 3 * cxxx * x**2]
        if self.dims == 2:
            x, y = k_v
            c, cx, cy, cxx, cxy, cyy, cxxx, cxxy, cxyy, cyyy = self.coef_j
            return [cx +
                    2 * cxx * x + cxy * y +
                    3 * cxxx * x**2 + 2 * cxxy * x * y + cxyy * y**2,
                    cy +
                    2 * cyy * y + cxy * x +
                    3 * cyyy * y**2 + 2 * cxyy * x * y + cxxy * x**2]
        1 / 0

    def hessian(self, k_v):
        if self.dims == 1:
            x = k_v[0]
            _, _, cxx, cxxx = self.coef_j
            return [[2 * cxx + 6 * cxxx * x]]
        if self.dims == 2:
            x, y = k_v
            _, _, _, cxx, cxy, cyy, cxxx, cxxy, cxyy, cyyy = self.coef_j
            return [[2 * cxx + 6 * cxxx * x + 2 * cxxy * y,
                     cxy + 2 * cxyy * y + 2 * cxxy * x],
                    [cxy + 2 * cxyy * y + 2 * cxxy * x,
                     2 * cyy + 6 * cyyy * y + 2 * cxyy * x]]
        1 / 0


if __name__ == '__main__':
    main()
