import pytest
from pathlib import Path


# ASE phonons recipe is deprecated and file handling is broken since ASR
# knows too much about ASE's own file handling.  So we just mark as xfail.
@pytest.mark.xfail
@pytest.mark.ci
def test_phonons(asr_tmpdir_w_params, mockgpaw, test_material, get_webcontent):
    """Simple test of phonon recipe."""
    from asr.phonons import main
    test_material.write('structure.json')
    main()
    p = Path('phonons.txt')
    assert p.is_file()
    text = p.read_text()
    assert '"xc": "PBE"' in text, p.read_text()

    content = get_webcontent()
    assert f"Phonons" in content, content
