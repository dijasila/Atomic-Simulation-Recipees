"""Effective masses."""
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
from ase.dft.bandgap import bandgap
from ase.units import Bohr, Ha

from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates
from gpaw.typing import Array1D, Array2D, Array3D


def fit(kpoints: Array1D,
        fermi_level: float,
        eigenvalues: Array2D,
        fingerprints: Array3D,
        spinprojections: Array3D = None,
        kind: str = 'cbm',
        N: int = 4,
        plot: bool = True) -> List[Tuple[float, float, float, Array1D]]:
    """...

    >>> k = np.linspace(-1, 1, 7)
    >>> eigs = 0.5 * k**2 * Ha * Bohr**2
    >>> minima = fit(kpoints=k,
    ...              fermi_level=-1.0,
    ...              eigenvalues=eigs[:, np.newaxis],
    ...              fingerprints=np.zeros((7, 1, 1)),
    ...              kind='cbm',
    ...              N=1,
    ...              plot=False)
    k [Ang^-1]  e-e_F [eV]    m [m_e]
        -0.000       1.000      1.000
    >>> k0, e0, m0 = minima[0]
    """

    K = len(kpoints)

    nocc = (eigenvalues[0] < fermi_level).sum()

    if kind == 'cbm':
        bands = slice(nocc, nocc + N)
        eigs = eigenvalues[:, bands] - fermi_level
    else:
        bands = slice(nocc - 1, nocc - 1 - N, -1)
        eigs = fermi_level - eigenvalues[:, bands]

    fps = fingerprints[:, bands]
    sps = spinprojections[:, bands]

    eigs2 = np.empty_like(eigs)
    fps2 = np.empty_like(fps)
    sps2 = np.empty_like(sps)
    imin = eigs[:, 0].argmin()
    i0 = K // 2
    for i in range(K):
        eigs2[(i0 + i) % K] = eigs[(imin + i) % K]
        fps2[(i0 + i) % K] = fps[(imin + i) % K]
        sps2[(i0 + i) % K] = sps[(imin + i) % K]
    x = kpoints[imin] + kpoints - kpoints[i0]

    eigs, sps = connect(eigs2, fps2, sps2)

    extrema = {}
    indices = eigs[i0].argsort()
    print('k [Ang^-1]  e-e_F [eV]    m [m_e]            spin [x,y,z]')
    for n in indices:
        band = eigs[:, n]
        i = band.argmin()
        if 2 <= i <= K - 3:
            poly = np.polyfit(x[i - 2:i + 3], band[i - 2:i + 3], 2)
            dx = 1.5 * (x[i + 2] - x[i])
            xfit = np.linspace(x[i] - dx, x[i] + dx, 61)
            yfit = np.polyval(poly, xfit)
            mass = 0.5 * Bohr**2 * Ha / poly[0]
            assert mass > 0
            k = -0.5 * poly[1] / poly[0]
            energy = np.polyval(poly, k)
            if kind == 'vbm':
                energy *= -1
                yfit *= -1
            spin = sps[i, n]
            print(f'{k:10.3f} {energy:11.3f} {mass:10.3f}',
                  '  (' + ', '.join(f'{s:+.2f}' for s in spin) + ')')
            extrema[n] = (xfit, yfit, k, energy, mass, spin)

    if kind == 'vbm':
        eigs *= -1

    if plot:
        import matplotlib.pyplot as plt
        color = 0
        for n in indices:
            plt.plot(x, eigs[:, n], 'o', color=f'C{color}')
            if n in extrema:
                xfit, yfit, *_ = extrema[n]
                plt.plot(xfit, yfit, '-', color=f'C{color}')
            color += 1
        plt.xlabel('k [Ang$^{-1}$]')
        plt.ylabel('e - e$_F$ [eV]')
        plt.show()

    return [(k, energy, mass, spin)
            for (_, _, k, energy, mass, spin) in extrema.values()]


def a_test():
    k = np.linspace(-1, 1, 7)
    b1 = (k - 0.2)**2
    b2 = 1 * (k + 0.2)**2 + 0.01 * 0
    eigs = np.array([b1, b2]).T
    indices = eigs.argsort(axis=1)
    eigs = np.take_along_axis(eigs, indices, axis=1)
    fps = np.zeros((7, 2, 2))
    fps[:, 0, 0] = 1
    fps[:, 1, 1] = 1
    fps[3] = 0.0
    fps[:, :, 0] = np.take_along_axis(fps[:, :, 0], indices, axis=1)
    fps[:, :, 1] = np.take_along_axis(fps[:, :, 1], indices, axis=1)
    eigs = fit(k, eigs, fps)
    import matplotlib.pyplot as plt
    plt.plot(k, eigs[:, 0])
    plt.plot(k, eigs[:, 1])
    plt.show()


def extract_stuff_from_gpw_file(gpwpath: Path,
                                soc: False,
                                outpath: Path = None) -> None:
    calc = GPAW(gpwpath)
    stuff = extract_stuff_from_gpaw_calculation(calc)
    outpath.write_bytes(pickle.dumps(stuff))


def extract_stuff_from_gpaw_calculation(calc,
                                        soc: False) -> Dict[str, Any]:
    kd = calc.wfs.kd
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
        eig_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                             for k in range(kd.nibzkpts)]
                            for s in range(kd.nspins)])
        proj_sknI = np.array([[calc.get_projections(kpt=k, spin=s)
                               for k in range(kd.nibzkpts)]
                              for s in range(kd.nspins)])
        spinproj_knv = states.spin_projections()

    K1, K2, K3 = tuple(kd.N_c)
    _, N, I = proj_knI.shape
    return {'cell': calc.atoms.cell,
            'kpts': k_kc.reshape((K1, K2, K3, 3)),
            'fermilevel': fermilevel,
            'eigs': eig_kn.reshape((K1, K2, K3, N)),
            'projs': proj_knI.reshape((K1, K2, K3, N, I)),
            'spinprojs': spinproj_knv.reshape((K1, K2, K3, N, 3))}


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


def main(kind='cbm', N=4):
    import sys
    path = Path(sys.argv[1])
    (k_ijkc, cell_cv, fermi_level, eig_ijkn, fp_ijknx) = \
        extract_stuff_from_gpw_file(path)
    print(k_ijkc.shape)
    nocc = (eig_ijkn[0, 0] < fermi_level).sum()
    print(nocc)

    if kind == 'cbm':
        bands = slice(nocc, nocc + N)
        eig_ijkn = eig_ijkn[..., bands] - fermi_level
    else:
        bands = slice(nocc - 1, nocc - 1 - N, -1)
        eig_ijkn = fermi_level - eig_ijkn[:, :, bands]

    fp_ijknx = fp_ijknx[..., bands, :]

    ijk = eig_ijkn[..., 0].ravel().argmin()
    i, j, k = np.unravel_index(ijk, eig_ijkn.shape[:3])
    print(i, j, k)
    print(k_ijkc[i, j, k])
    print(eig_ijkn[i, j, k])
    import matplotlib.pyplot as plt
    plt.plot(eig_ijkn[i, :, 0, 0])
    plt.plot(eig_ijkn[i, :, 0, 1])
    plt.show()
    I = np.arange(i - 3, i + 4)
    J = np.arange(j - 3, j + 4)
    K = np.arange(k - 0, k + 1)
    eig_ijkn = eig_ijkn[I][:, J][:, :, K]
    b_ijkn = connect(fp_ijknx[I][:, J][:, :, K])
    k_ijkc = np.indices((7, 7, 1)).transpose((1, 2, 3, 0)) - (3, 3, 0)
    print(k_ijkc.shape)
    k_ijkv = k_ijkc# @ np.linalg.inv(cell_cv).T * 2 * pi

    f_ijkn = np.zeros_like(eig_ijkn) + np.nan
    import matplotlib.pyplot as plt
    for b in range(N):
        f_ijkn[(b_ijkn == b).any(3), b] = eig_ijkn[b_ijkn == b]
        # x = f_ijkn[:, 3, 0, b]
        # plt.plot(x, label=str(b))
    # plt.legend()
    # plt.show()

    for b in range(N):
        mask_ijkn = b_ijkn == b
        eig_k = eig_ijkn[mask_ijkn]
        k_kv = k_ijkv[mask_ijkn.any(axis=3)]
        k = (k_kv**2).sum(1).argsort()[:15]
        k_kv = k_kv[k]
        eig_k = eig_k[k]


class Fit3D:
    def __init__(self, k_iv, eig_i):
        self.dims = k_iv.shape[1]
        if self.dims == 1:
            x = k_iv[:, 0]
            f_ji = np.array([x**0,
                             x,
                             x**2,
                             x**3])
        elif self.dims == 2:
            x, y = k_iv.T
            f_ji = np.array([x**0,
                             x,
                             y,
                             x**2,
                             y**2,
                             x * y,
                             x**3,
                             y**3,
                             x**2 * y,
                             y**2 * x])
        else:
            x, y, z = k_iv.T
            f_ji = np.array([x**0,
                             x,
                             y,
                             z,
                             x**2,
                             y**2,
                             z**2,
                             x * y,
                             y * z,
                             z * x,
                             x**3,
                             y**3,
                             z**3,
                             x**2 * y,
                             x**2 * z,
                             y**2 * x,
                             y**2 * z,
                             z**2 * x,
                             z**2 * y,
                             x * y * z])
        self.coef_j = np.linalg.solv(f_ji @ f_ji.T, f_ji @ eig_i)

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

    def find_minimum(self, k_v):
        from scipy.optimize import minimize

        def f(k_v):
            return self.value(k_v), self.gradient(k_v)

        o = minimize(f, [0, 0], jac=True)
        print(o)


if __name__ == '__main__':
    main()
