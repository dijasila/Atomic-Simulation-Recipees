from types import SimpleNamespace
from math import pi
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import GPAW

from asr.emasses2 import (_main, connect,
                          extract_soc_stuff_from_gpaw_calculation, fit)


def test_1d():
    """Two crossing bands with degeneracy."""
    a = 2.2
    m = 0.3
    k0 = 0.2
    k_i = np.linspace(-pi / a, pi / a, 7)
    b1_i = (k_i - k0)**2 / (2 * m) * Ha * Bohr**2
    b2_i = (k_i + k0)**2 / (2 * m) * Ha * Bohr**2
    eig_ijkn = np.array([b1_i, b2_i]).T.reshape((7, 1, 1, 2))
    fp_ijknx = np.zeros((7, 1, 1, 2, 2))
    fp_ijknx[:, :, :, 0, 0] = 1
    fp_ijknx[:, :, :, 1, 1] = 1
    n_ijkn = eig_ijkn.argsort(axis=3)
    eig_ijkn = np.take_along_axis(eig_ijkn, n_ijkn, axis=3)
    fp_ijknx = np.take_along_axis(fp_ijknx, n_ijkn[:, :, :, :, np.newaxis], axis=3)
    print(eig_ijkn.shape)
    print(eig_ijkn)
    connect(eig_ijkn, fp_ijknx)
    for n in [0, 1]:
        r = fit(k_i[:, np.newaxis] * (a / (2 * pi)),
                eig_ijkn[:, 0, 0, n],
                np.ones((1, 1)) * a)
        k_v, emin, mass_w, evec_wv, error_i = r
        print(r)
        assert k_v[0] == pytest.approx(-k0)
        assert emin == pytest.approx(0.0)
        assert mass_w[0] == pytest.approx(m)
        assert abs(error_i).max() == pytest.approx(0.0)
        k0 *= -1


def test_3d():
    """k_kc = np.zeros((10, 3))
    k_kc[:, 2] = np.linspace(-0.5, 0.5, 10, False)
    kpt_qs = [
        [SN(projections=SN(collect=lambda: np.eye(3)))
         for _ in range(2)]
        for _ in range(10)]
    c = 0.5 * Ha * Bohr**2 * (np.pi / 5)**2  # make mass = 1 m_e
    calc = SN(
        world=SN(size=1),
        wfs=SN(
            kd=SN(
                N_c=np.array([1, 1, 10]),
                bzk_kc=k_kc,
                ibzk_kc=k_kc,
                nspins=2,
                nbzkpts=10,
                nibzkpts=10),
            kpt_qs=kpt_qs),
        get_eigenvalues=lambda kpt, spin: [-c * (kpt - 5)**2,
                                           1 + c * (kpt - 5)**2,
                                           2],
        atoms=SN(
            cell=np.eye(3)),
        get_fermi_level=lambda: 0.5)
    """
    return SimpleNamespace()


def test_emass_h2():
    h2 = Atoms('H2',
               [[0, 0, 0], [0, 0, 0.74]],
               cell=[2.0, 3.0, 3.0],
               pbc=True)
    h2.calc = GPAW(mode={'name': 'pw',
                         'ecut': 300},
                   convergence={'bands': 2},
                   kpts=[20, 1, 1],
                   txt=None)
    h2.get_potential_energy()
    K_ijkc, eig_ijkn, proj_ijknI, spinproj_ijknv = \
        extract_soc_stuff_from_gpaw_calculation(h2.calc)
    print(eig_ijkn.shape)
    print(eig_ijkn[0, 0, 0])
    cell_cv = h2.calc.atoms.cell

    extrema = _main(cell_cv, K_ijkc, eig_ijkn, proj_ijknI)
    print(extrema)
