import pytest
from asr.c2db.polarizability import main
from asr.c2db.gs import calculate as gscalculate


@pytest.mark.ci
def test_polarizability(asr_tmpdir_w_params, mockgpaw, test_material,
                        get_webcontent, fast_calc):
    gsresult = gscalculate(atoms=test_material, calculator=fast_calc)
    main(gsresult=gsresult, kptdensity=2)
    test_material.write('structure.json')
    # content = get_webcontent()
    # assert "polarizability" in content, content
