from .conftest import test_materials, get_webcontent
import pytest


@pytest.mark.parametrize("atoms", test_materials)
def test_projected_bandstructure(separate_folder, mockgpaw, atoms):
    from asr.projected_bandstructure import main
    atoms.write("structure.json")
    main()

    get_webcontent()
