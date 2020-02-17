from .conftest import test_materials
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_bandstructure_main(separate_folder, usemocks, fs, atoms):
    from ase.io import write
    from asr.bandstructure import main
    write('structure.json', atoms)
    main()
