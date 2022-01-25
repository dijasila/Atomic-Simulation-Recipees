import pytest


@pytest.mark.ci
def test_g_factors_from_gyromagnetic_ratios():
    from asr.hyperfine import (g_factors_from_gyromagnetic_ratios,
                               gyromagnetic_ratios)

    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)
    print(g_factors, gyromagnetic_ratios)
    assert False
