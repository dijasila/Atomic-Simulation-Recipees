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
    from .materials import Ag, BN
    from ase.io import write
    from asr.hyperfine import main
    from asr.gs import calculate

    atoms = BN.copy()
    write('structure.json', atoms)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"size": (1, 1, 1), "gamma": True},
        },
    )

    # recipe should raise an error for zero magnetic moment
    try:
        # calculate_hyperfine(atoms, calc)
        main()
        calculated = True
    except AssertionError:
        calculated = False

    assert not calculated

    # set magnetic moment
    atoms = Ag.copy()
    write('structure.json', atoms)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"size": (1, 1, 1), "gamma": True},
        },
    )
    res = main()

    # HF interaction energy and sct are not implemented yet
    res_ag = 39.878123
    assert res['delta_E_hyp'] is None
    assert res['sc_time'] is None
    for eigenvalue in res['hyperfine'][0]['eigenvalues']:
        assert eigenvalue == pytest.approx(res_ag)
