from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_database_crosslinks(asr_tmpdir, crosslinks_test_dbs):

