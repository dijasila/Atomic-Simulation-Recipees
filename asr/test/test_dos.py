import pytest
from asr.c2db.dos import main


@pytest.mark.ci
@pytest.mark.parallel
def test_dos(asr_tmpdir_w_params, mockgpaw, test_material, get_webcontent):
    main(atoms=test_material, kptdensity=2)
    test_material.write("structure.json")
    get_webcontent()
