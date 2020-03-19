import pytest
from .conftest import test_materials


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_gs(separate_folder, mockgpaw, atoms):
    from asr.phonons import main
    atoms.write('structure.json')
    main()
