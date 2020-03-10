import pytest
from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_polarizability(separate_folder, mockgpaw, atoms):
    from asr.polarizability import main
    atoms.write('structure.json')
    main()
    content = get_webcontent()
    assert "polarizability" in content, content
