import pytest
from ase.io import write
from asr.bandstructure import main
from asr.structureinfo import main as structinfo


@pytest.mark.ci
def test_bandstructure_main(asr_tmpdir_w_params, mockgpaw, test_material,
                            get_webcontent):

    # XXX: Structureinfo is needed for the webpanel to function.
    # This is not really a standard dependency and it should probably
    # be fixed in the future.
    structinfo(atoms=test_material)

    main(atoms=test_material)

    write('structure.json', test_material)
    get_webcontent()
