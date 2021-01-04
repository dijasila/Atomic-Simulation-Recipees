import pytest
from asr.polarizability import main


@pytest.mark.ci
def test_polarizability(asr_tmpdir_w_params, mockgpaw, test_material,
                        get_webcontent):
    main(atoms=test_material)
    test_material.write('structure.json')
    content = get_webcontent()
    assert "polarizability" in content, content
