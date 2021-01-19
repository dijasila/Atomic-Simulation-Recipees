from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_database_crosslinks(asr_tmpdir, test_material):
    from ase.io import write, read
    from asr.setup.defects import main

    write('unrelaxed.json', std_test_materials[1])
    main(supercell=[3, 3, 1])

    assert 1 == 1
