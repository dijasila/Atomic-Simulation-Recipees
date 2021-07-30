import pytest
from asr.c2db.projected_bandstructure import main


@pytest.mark.ci
def test_projected_bs_mocked(asr_tmpdir, mockgpaw, get_webcontent,
                             test_material):
    main(atoms=test_material)
    test_material.write("structure.json")
    get_webcontent()
