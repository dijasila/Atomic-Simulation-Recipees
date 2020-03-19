import pytest
from .conftest import test_materials


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_borncharges(separate_folder, mockgpaw, atoms):
    from asr.formalpolarization import main
    atoms.write('structure.json')
    main()
