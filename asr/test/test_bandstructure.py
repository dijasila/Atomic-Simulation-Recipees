import pytest


@pytest.mark.ci
def test_bandstructure_main(separate_folder, mockgpaw, test_material):
    from ase.io import write
    from asr.bandstructure import main
    write('structure.json', test_material)
    main()
