import pytest


@pytest.mark.ci
def test_bandstructure_main(asr_tmpdir_w_params, mockgpaw, test_material,
                            get_webcontent):
    from ase.io import write
    from asr.bandstructure import main
    write('structure.json', test_material)
    main()
    get_webcontent()
