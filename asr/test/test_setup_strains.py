import pytest


@pytest.mark.ci
@pytest.mark.parametrize("pbc", [[True, ] * 3,
                                 [True, True, False],
                                 [False, False, True]])
def test_setup_strains_get_relevant_strains(asr_tmpdir_w_params, pbc):
    from asr.setup.strains import get_relevant_strains

    ij = set(get_relevant_strains(pbc))
    if sum(pbc) == 3:
        ij2 = {(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
    elif sum(pbc) == 2:
        ij2 = {(0, 0), (1, 1), (0, 1)}
    elif sum(pbc) == 1:
        ij2 = {(2, 2)}

    assert ij == ij2


@pytest.mark.ci
def test_setup_strains(asr_tmpdir_w_params, mockgpaw, test_material):
    from asr.setup.strains import main
    main(
        atoms=test_material,
        strain_percent=1,
        i=0,
        j=1)
