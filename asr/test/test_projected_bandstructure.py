import pytest


@pytest.mark.ci
def test_projected_bandstructure(asr_tmpdir_w_params, mockgpaw, test_material,
                                 get_webcontent):
    from asr.projected_bandstructure import main
    test_material.write("structure.json")
    main()

    get_webcontent()
