import pytest


@pytest.mark.ci
def test_phonopy(asr_tmpdir_w_params, mockgpaw, get_webcontent):
    """Simple test of phononpy recipe."""
    from asr.phonopy import main
    from ase.build import bulk

    N = 2

    atoms = bulk('Al', 'fcc', a=4.05)

    data = main(atoms=atoms, sc=[N, N, N], calculator={'name': 'emt'})
    assert data['minhessianeig'] == pytest.approx(0)
    assert data['dynamic_stability_level'] == 3
