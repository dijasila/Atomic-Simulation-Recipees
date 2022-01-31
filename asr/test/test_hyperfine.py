import pytest
from .materials import std_test_materials


@pytest.mark.ci
def test_g_factors_from_gyromagnetic_ratios():
    from asr.hyperfine import (g_factors_from_gyromagnetic_ratios,
                               gyromagnetic_ratios)

    result_hydrogen = 5.5856946
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    assert g_factors['H'] == pytest.approx(result_hydrogen)


@pytest.mark.ci
def test_paired_system(asr_tmpdir):
    from .materials import BN
    from asr.hyperfine import (gyromagnetic_ratios,
                               rescale_hyperfine_tensor,
                               g_factors_from_gyromagnetic_ratios)
    import numpy as np

    atoms = BN.copy()
    refarray = np.ones((len(atoms), 3, 3))
    symbols = atoms.symbols
    magmoms = np.zeros(len(atoms))
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)
    # recipe should raise an error for zero magnetic moment
    try:
        rescale_hyperfine_tensor(refarray, g_factors, symbols, magmoms)
        calculated = True
    except AssertionError:
        calculated = False

    assert not calculated

    # run the rescaling function for non-zero magnetic_moment
    magmoms = np.ones(len(atoms))
    res_hf, used = rescale_hyperfine_tensor(
        refarray, g_factors, symbols, magmoms)

    # compare HF eigenvalues
    for i, hfres in enumerate(res_hf):
        assert hfres['kind'] == symbols[i]
        assert hfres['magmom'] == pytest.approx(1)


@pytest.mark.parametrize('atoms', std_test_materials)
@pytest.mark.ci
def test_get_gyro_results(atoms):
    from asr.hyperfine import (gyromagnetic_ratios,
                               rescale_hyperfine_tensor,
                               g_factors_from_gyromagnetic_ratios,
                               get_gyro_results)
    import numpy as np

    refarray = np.ones((len(atoms), 3, 3))
    symbols = atoms.symbols
    magmoms = np.ones(len(atoms))
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    _, used = rescale_hyperfine_tensor(
        refarray, g_factors, symbols, magmoms)

    res_gyro = get_gyro_results(used)
    for res in res_gyro:
        symbol = res['symbol']
        assert symbol in symbols
        assert g_factors[symbol] == res['g']
