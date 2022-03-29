import pytest
import numpy as np
from .materials import std_test_materials, BN
from asr.hyperfine import (g_factors_from_gyromagnetic_ratios,
                           gyromagnetic_ratios,
                           rescale_hyperfine_tensor,
                           HyperfineNotCalculatedError,
                           get_atoms_close_to_center,
                           GyromagneticResult)


@pytest.mark.ci
def test_g_factors_from_gyromagnetic_ratios():
    result_hydrogen = 5.5856946
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    assert g_factors['H'] == pytest.approx(result_hydrogen)


@pytest.mark.ci
def test_paired_system(asr_tmpdir):
    atoms = BN.copy()
    refarray = np.ones((len(atoms), 3, 3))
    symbols = atoms.symbols
    magmoms = np.zeros(len(atoms))
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)
    # recipe should raise an error for zero magnetic moment
    with pytest.raises(HyperfineNotCalculatedError):
        rescale_hyperfine_tensor(refarray, g_factors, symbols, magmoms)

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
    refarray = np.ones((len(atoms), 3, 3))
    symbols = atoms.symbols
    magmoms = np.ones(len(atoms))
    g_factors = g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios)

    _, used = rescale_hyperfine_tensor(
        refarray, g_factors, symbols, magmoms)

    res_gyro = GyromagneticResult.fromdict(used)
    for res in res_gyro:
        symbol = res['symbol']
        assert symbol in symbols
        assert g_factors[symbol] == res['g']


@pytest.mark.ci
def test_get_atoms_close_to_center():
    atoms = BN.copy()
    z = atoms.cell.lengths()[2] / 2.
    supercell = atoms.repeat((4, 4, 1))
    indices, distances = get_atoms_close_to_center([0, 0, z], supercell)
    assert distances[0] == pytest.approx(0)
