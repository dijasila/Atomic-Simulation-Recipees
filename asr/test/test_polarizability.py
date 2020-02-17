import pytest
from .conftest import test_materials


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_polarizability(separate_folder, usemocks, atoms):
    from asr.polarizability import main
    atoms.write('structure.json')
    main()
