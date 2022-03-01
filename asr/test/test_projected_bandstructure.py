import pytest


@pytest.mark.xfail(reason='TODO')
@pytest.mark.ci
def test_projected_bs_mocked(asr_tmpdir, mockgpaw, get_webcontent,
                             test_material):
    from asr.c2db.projected_bandstructure import main
    main(atoms=test_material)
    #test_material.write("structure.json")
    #get_webcontent()
