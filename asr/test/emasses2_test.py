import pytest
from types import SimpleNamespace as SN
from ase.units import Bohr, Ha
from ase import Atoms
import numpy as np

from asr.emasses2 import (connect, extract_soc_stuff_from_gpaw_calculation,
                          find_minima, fit)
from gpaw import GPAW


def test_1d():
    """Two crossing bands with degeneracy."""
    k = np.linspace(-1, 1, 7)
    b1 = (k - 0.2)**2
    b2 = 1 * (k + 0.2)**2 + 0.01 * 0
    eigs = np.array([b1, b2]).T
    indices = eigs.argsort(axis=1)
    eigs = np.take_along_axis(eigs, indices, axis=1)
    fps = np.zeros((7, 2, 2))
    fps[:, 0, 0] = 1
    fps[:, 1, 1] = 1
    # fps[3] = 0.0
    fps[:, :, 0] = np.take_along_axis(fps[:, :, 0], indices, axis=1)
    fps[:, :, 1] = np.take_along_axis(fps[:, :, 1], indices, axis=1)
    indices = connect(fps.reshape((7, 1, 1, 2, 2)))
    assert indices[:, 0, 0].tolist() == [[0, 1], [0, 1], [0, 1],
                                         [1, 0], [1, 0], [1, 0], [1, 0]]
    band0 = eigs[indices[:, 0, 0] == 0]
    band1 = eigs[indices[:, 0, 0] == 1]
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(k, band0)
        plt.plot(k, band1)
        plt.show()


def test_extract_stuff_from_gpaw_calculation():
    k_kc = np.zeros((10, 3))
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
    dct = extract_soc_stuff_from_gpaw_calculation(calc)
    print(dct)
    bands = connect(dct['proj_ijknI'])
    print(bands[0, 0].tolist())
    assert bands[0, 0].tolist() == [[0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 4, 5, 2, 3],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5],
                                    [0, 1, 2, 3, 4, 5]]

    bands = find_minima(**dct)
    for band in bands[:2]:
        print(band)
        assert (band[1] == 2.0).all()
        with pytest.raises(ValueError):
            fit(*band)
    for band in bands[2:]:
        print(band)
        k_v, e, m_v, h_vv = fit(*band)
        assert k_v[0] == pytest.approx(0.0)
        assert e == pytest.approx(1.0)
        assert m_v[0] == pytest.approx(1.0)
        assert h_vv[0, 0] == pytest.approx(1.0)


def test_emass_h2():
    h2 = Atoms('H2', [[0, 0, 0], [0, 0, 0.74]],
               cell=[2, 3, 3], pbc=True)
    h2.calc = GPAW(mode={'name': 'pw', 'ecut': 300},
                   kpts=[20, 1, 1],
                   txt=None)
    h2.get_potential_energy()
    dct = extract_soc_stuff_from_gpaw_calculation(h2.calc)
    # print(dct)
    bands = find_minima(**dct, kind='vbm')
    print(bands)
    for band in bands:
        fit(*band)
