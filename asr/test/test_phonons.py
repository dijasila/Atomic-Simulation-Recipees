import pytest
from .conftest import test_materials


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_phonons(separate_folder, mockgpaw, atoms):
    """Simple test of phonon recipe."""
    from asr.core import read_json
    from asr.phonons import main
    atoms.write('structure.json')
    main()

    calc = read_json('gs.gpw')
    assert calc['xc'] == 'PBE'
