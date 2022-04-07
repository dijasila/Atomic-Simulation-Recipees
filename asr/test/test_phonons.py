import pytest


@pytest.mark.ci
def test_phonons(asr_tmpdir_w_params, mockgpaw, test_material, get_webcontent):
    """Simple test of phonon recipe."""
    from asr.c2db.phonons import main
    test_material.write('structure.json')
    main(atoms=test_material)
    content = get_webcontent()
    assert "Phonons" in content, content
