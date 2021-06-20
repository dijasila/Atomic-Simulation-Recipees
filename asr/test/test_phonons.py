import pytest


# ASE phonons recipe is deprecated and file handling is broken since ASR
# knows too much about ASE's own file handling.  So we just mark as xfail.
@pytest.mark.ci
def test_phonons(asr_tmpdir_w_params, mockgpaw, test_material, get_webcontent):
    """Simple test of phonon recipe."""
    from asr.phonons import main
    test_material.write('structure.json')
    main(atoms=test_material)
    content = get_webcontent()
    assert "Phonons" in content, content
