import pytest


@pytest.mark.ci
def test_g_factors_from_gyromagnetic_ratios():
    from asr.hyperfine import (g_factors_from_gyromagnetic_ratios,
                               gyromagnetic_ratios)

    result_hydrogen = 5.5856946
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    assert g_factors['H'] == pytest.approx(result_hydrogen)


@pytest.mark.ci
def test_paired_system(asr_tmpdir):
    from gpaw import GPAW
    from .materials import Ag
    from asr.hyperfine import calculate_hyperfine
    import numpy as np

    atoms = Ag.copy()

    calc = GPAW(txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()

    # recipe should raise an error for zero magnetic moment
    try:
        calculate_hyperfine(atoms, calc)
        calculated = True
    except AssertionError:
        calculated = False

    assert not calculated

    # set magnetic moment
    atoms.set_initial_magnetic_moments(np.ones(len(atoms)))
    hf_res, gyro_res, hf_int, sct = calculate_hyperfine(atoms, calc)

    # HF interaction energy and sct are not implemented yet
    res_ag = 39.878123
    assert hf_int is None
    assert sct is None
    for eigenvalue in hf_res[0]['eigenvalues']:
        assert eigenvalue == pytest.approx(res_ag)
