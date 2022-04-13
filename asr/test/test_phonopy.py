import pytest


@pytest.mark.ci
def test_phonopy(mockgpaw, get_webcontent, in_tempdir):
    """Simple test of phononpy recipe."""
    from asr.c2db.phonopy import PhonopyWorkflow
    from ase.build import bulk

    N = 2

    atoms = bulk('Al', 'fcc', a=4.05)

    phonons = PhonopyWorkflow(
        atoms=atoms,
        calculator={'name': 'emt'},
        sc=[N, N, N])

    data = phonons.post
    assert data['minhessianeig'] == pytest.approx(0)
    assert data['dynamic_stability_level'] == 3
