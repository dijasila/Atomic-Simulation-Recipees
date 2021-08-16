from pathlib import Path

import pytest
import numpy as np

from asr.c2db.infraredpolarizability import create_plot_simple


@pytest.mark.ci
@pytest.mark.parametrize('ndim', range(0, 4))
def test_plot(asr_tmpdir, ndim):
    nw = 40
    omega_w = np.linspace(0, 10, nw)
    fname = create_plot_simple(
        ndim=ndim,
        omega_w=omega_w,
        fname='thefile.png',
        maxomega=omega_w[-1] * 1.2,
        alpha_w=np.exp(1j * omega_w),
        alphavv_w=0.1 * np.cos(omega_w),
        axisname='x',
        omegatmp_w=omega_w.copy())
    assert Path(fname).is_file()  # XXX not a very thorough test.
