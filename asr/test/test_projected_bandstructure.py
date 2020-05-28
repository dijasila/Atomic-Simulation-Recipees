import pytest


@pytest.mark.ci
def test_projected_bs_mocked(asr_tmpdir, mockgpaw, get_webcontent,
                             test_material):
    from asr.projected_bandstructure import main
    test_material.write("structure.json")
    main()

    get_webcontent()
