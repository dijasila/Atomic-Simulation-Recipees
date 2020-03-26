import pytest
from .conftest import test_materials
from pathlib import Path


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_phonons(separate_folder, mockgpaw, atoms):
    """Simple test of phonon recipe."""
    from asr.phonons import main
    atoms.write('structure.json')
    main()
    p = Path('phonons.txt')
    assert p.is_file()
    text = p.read_text()
    assert '"xc": "PBE"' in text, p.read_text()
