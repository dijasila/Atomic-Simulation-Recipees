import pytest


@pytest.mark.ci
def test_projected_bs_mocked(asr_tmpdir, mockgpaw, get_webcontent,
                             test_material, fast_calc):
    from asr.c2db.gs import GS
    from asr.c2db.bandstructure import BS
    from asr.c2db.projected_bandstructure import main

    gs = GS(atoms=test_material, calculator=fast_calc)
    bs = BS(gs=gs, npoints=10)
    main(bscalculateresult=bs.calculateresult)
    # main(atoms=test_material)
    # BS(atoms=test_material, calculator=fast_calc)
    #test_material.write("structure.json")
    #get_webcontent()
