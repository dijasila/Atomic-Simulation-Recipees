import pytest


@pytest.mark.ci
def test_g_factors_from_gyromagnetic_ratios():
    from asr.hyperfine import (g_factors_from_gyromagnetic_ratios,
                               gyromagnetic_ratios)

    result_hydrogen = 5.5856946
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    assert g_factors['H'] == pytest.approx(result_hydrogen)
