import pytest


@pytest.mark.ci
def test_phonopy(repo, mockgpaw, get_webcontent):
    """Simple test of phononpy recipe."""
    from asr.c2db.phonopy import PhonopyWorkflow
    from ase.build import bulk

    N = 2

    atoms = bulk('Al', 'fcc', a=4.05)

    phonons = repo.run_workflow_blocking(
        PhonopyWorkflow,
        atoms=atoms,
        calculator={'name': 'emt'},
        sc=[N, N, N])

    with repo:
        data = phonons.postprocess.value().output
    assert data['minhessianeig'] == pytest.approx(0)
    assert data['dynamic_stability_level'] == 3
