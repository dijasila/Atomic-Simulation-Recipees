from pathlib import Path

import pytest
import numpy as np

from asr.c2db.infraredpolarizability import (
    create_plot_simple, InfraredPolarizabilityWorkflow)
from asr.c2db.phonons import PhononWorkflow


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


def test_infrared_polarizability(repo, mockgpaw, asr_tmpdir, test_material,
                                 fast_calc):
    print(repo.root)

    phw = repo.run_workflow_blocking(
        PhononWorkflow,
        atoms=test_material,
        calculator=fast_calc)

    def gscalculate(rn):
        return rn.task('asr.c2db.gs.calculate', atoms=test_material,
                       calculator=fast_calc)

    # There is some room for improved interface here
    gs = repo.run_workflow_blocking(gscalculate)

    ipw = repo.run_workflow_blocking(
        InfraredPolarizabilityWorkflow,
        atoms=test_material,
        phonons=phw.postprocess.output,
        polarizability_gs=gs.output,
        borncalculator=fast_calc)
