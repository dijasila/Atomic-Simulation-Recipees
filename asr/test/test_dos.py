import pytest
from asr.c2db.dos import main


@pytest.fixture
def gsresult(asr_tmpdir_w_params, mockgpaw, test_material, fast_calc):
    from asr.c2db.gs import calculate
    return calculate(atoms=test_material, calculator=fast_calc)


@pytest.mark.ci
def test_dos(gsresult, get_webcontent):
    main(gsresult=gsresult, kptdensity=2)
    # test_material.write("structure.json")
    # get_webcontent()
