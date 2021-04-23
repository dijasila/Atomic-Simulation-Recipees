from pathlib import Path
import numpy as np


def test_plot(asr_tmpdir):
    from asr.infraredpolarizability import create_plot_simple

    nw = 40
    omega_w = np.linspace(0, 10, nw)
    fname = create_plot_simple(
        ndim=3,
        omega_w=omega_w,
        fname='thefile.png',
        maxomega=30,
        alpha_w=np.exp(1j * omega_w),
        alphavv_w=0.1 * np.cos(omega_w),
        axisname='x',
        omegatmp_w=omega_w.copy())
    assert Path(fname).is_file()  # XXX not a very thorough test.
