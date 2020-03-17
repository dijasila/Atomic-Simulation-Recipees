import pytest
from .conftest import test_materials, get_webcontent


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_formalpolarization(separate_folder, mockgpaw, atoms):
    from asr.formalpolarization import main
    atoms.write('structure.json')
    results = main()

    print(results)
    assert False
