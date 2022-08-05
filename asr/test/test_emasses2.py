from math import pi
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from asr.emasses2 import (EMassesResult, connect,
                          extract_soc_stuff_from_gpaw_calculation, find_mass,
                          fit_band, mass_plots, webpanel)


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
    fp_ijknx = np.take_along_axis(fp_ijknx, n_ijkn[:, :, :, :, np.newaxis],
                                  axis=3)
    eig_ijkn = connect(eig_ijkn, fp_ijknx)
    for n in [0, 1]:
        r = fit_band(k_i[:, np.newaxis],
                     eig_ijkn[:, 0, 0, n])
        k_v, emin, mass_w, evec_wv, error_i, fit = r
        assert k_v[0] == pytest.approx(-k0)
        assert emin == pytest.approx(0.0)
        assert mass_w[0] == pytest.approx(m)
        assert abs(error_i).max() == pytest.approx(0.0)
        k0 *= -1


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_emass_h2():
    from gpaw import GPAW
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
    data = extract_soc_stuff_from_gpaw_calculation(h2.calc)

    extrema = []
    for kind in ['vbm', 'cbm']:
        massdata = find_mass(**data[kind], kind=kind)
        extrema.append(massdata)

    print(extrema)
    vbm, cbm = extrema
    result = EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)
    row = SimpleNamespace(data={'results-asr.emasses2.json': result})

    mass_plots(row, 'cbm.png', 'vbm.png')
    print(webpanel(result, row, {}))
