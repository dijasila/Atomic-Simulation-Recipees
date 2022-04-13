import pytest
from ase.utils import workdir


@pytest.mark.ci
def test_phonons(repo, asr_tmpdir_w_params, mockgpaw, test_material,
                 get_webcontent, fast_calc):
    """Simple test of phonon recipe."""
    from asr.c2db.phonons import PhononWorkflow
    test_material.write('structure.json')

    repo.run_workflow_blocking(
        PhononWorkflow,
        atoms=test_material, calculator=fast_calc)

    # Need to explicitly change paths or the phonon caches will clash

    # main(atoms=test_material)
    # content = get_webcontent()
    # assert "Phonons" in content, content
