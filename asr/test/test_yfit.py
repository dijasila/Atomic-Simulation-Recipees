import numpy as np
import pytest

from asr.utils.mass_fit import Y, YFunctions


def test_cubical_harmonics_orthogonality():
    N = 100
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.exp(1j * theta)
    x = z.real
    y = z.imag
    M = []
    for l, m in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]:
        M.append(Y(l, m, x, y))
    M = np.array(M)
    E = M @ M.T / N - np.eye(5)
    assert abs(E).max() < 1e-10


@pytest.mark.parametrize('ndims', [1, 2, 3])
def test_y_fit(ndims):
    if ndims == 1:
        k_iv = np.indices((5,)).reshape((1, 5)).T - 2
    elif ndims == 2:
        k_iv = np.indices((5, 5)).reshape((2, 25)).T - 2
    else:
        k_iv = np.indices((5, 5, 5)).reshape((3, 125)).T - 2
    e_i = (k_iv**2).sum(1)
    fit, error = YFunctions(ndims, 4).fit_data(k_iv, e_i)
    assert abs(error) < 1e-11
    assert not fit.kmin_v.any()
    assert abs(fit.emin) < 1e-11
    assert abs(fit.hessian() - 2 * np.eye(ndims)).max() < 1e-12


def test_warp():
    k_iv = np.indices((3, 3)).reshape((2, 9)).T
    e_i = np.ones(9)
    e_i[4] = 0.0
    fit, error = YFunctions(2, 4).fit_data(k_iv, e_i)
    print(error, fit.kmin_v, fit.emin, fit.coef_j)
    assert abs(error) < 1e-13
    assert (fit.kmin_v == 1).all()
    assert abs(fit.emin) < 1e-11
    assert fit.warping() == pytest.approx(1 / 18)
