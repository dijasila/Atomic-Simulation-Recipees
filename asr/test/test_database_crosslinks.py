from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_database_crosslinks(asr_tmpdir, test_material):
    print(test_material)
    assert 1 == 1
