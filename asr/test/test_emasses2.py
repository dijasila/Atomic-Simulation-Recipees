import os
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from asr.emasses2 import (EMassesResult, GPAWEigenvalueCalculator, _main,
                          fit_band, mass_plots, webpanel)


def test_1d_mass():
    a = 2.2
    m = 0.3
    k0 = 0.2
    k_i = np.linspace(-pi / a, pi / a, 7)
    eig_i = (k_i - k0)**2 / (2 * m) * Ha * Bohr**2
    k_v, emin, mass_w, evec_wv, error_i, fit = fit_band(k_i[:, np.newaxis],
                                                        eig_i)
    assert k_v[0] == pytest.approx(k0)
    assert emin == pytest.approx(0.0)
    assert mass_w[0] == pytest.approx(m)
    assert abs(error_i).max() == pytest.approx(0.0)


def e(k):
    """Two crossing bands with degeneracy."""
    m = 0.3
    k0 = 0.2
    return min((k - k0)**2, (k + k0)**2) / (2 * m) * Ha * Bohr**2


class EigCalc:
    cell_cv = np.diag([2.2, 10, 20])

    def get_band(self, kind):
        a = self.cell_cv[0, 0]
        k_i = np.linspace(-pi / a, pi / a, 7, False)
        k_ijkv = np.zeros((7, 1, 1, 3))
        k_ijkv[:, 0, 0, 0] = k_i
        return k_ijkv, np.array([e(k) for k in k_i]).reshape((7, 1, 1))

    def get_new_band(self, kind, kpt_xv):
        return np.array([e(k) for k in kpt_xv[:, 0]])


def test_1d_mass2():
    """Two crossing bands with degeneracy."""
    eigcalc = EigCalc()
    _main(eigcalc, 'cbm', 4)


@pytest.fixture
def h2_calc():
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
    return h2.calc


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
@pytest.mark.parametrize('angles', [(None, None), (0.0, 0.0)])
def test_emass_h2(tmp_path, h2_calc, angles):
    os.chdir(tmp_path)
    eigcalc = GPAWEigenvalueCalculator(h2_calc, *angles)

    extrema = []
    for kind in ['vbm', 'cbm']:
        massdata = _main(eigcalc, kind)
        extrema.append(massdata)

    print(extrema)
    vbm, cbm = extrema

    assert cbm['energy'] - vbm['energy'] == pytest.approx(10.8, abs=0.1)
    assert abs(vbm['k_v'][0]) == pytest.approx(pi / 2, abs=0.005)
    assert abs(cbm['k_v'][0]) == pytest.approx(pi / 2, abs=0.005)
    assert abs(vbm['mass_w'][0]) == pytest.approx(0.48, abs=0.01)
    assert abs(cbm['mass_w'][0]) == pytest.approx(0.32, abs=0.01)

    result = EMassesResult.fromdata(vbm_mass=vbm, cbm_mass=cbm)
    row = SimpleNamespace(data={'results-asr.emasses2.json': result})

    mass_plots(row, 'cbm.png', 'vbm.png')
    print(webpanel(result, row, {}))
