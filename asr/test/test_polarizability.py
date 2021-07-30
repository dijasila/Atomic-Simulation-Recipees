import pytest
from asr.c2db.polarizability import main


@pytest.mark.ci
def test_polarizability(asr_tmpdir_w_params, mockgpaw, test_material,
                        get_webcontent, fast_calc):
    main(atoms=test_material, calculator=fast_calc, kptdensity=2)
    test_material.write('structure.json')
    content = get_webcontent()
    assert "polarizability" in content, content
