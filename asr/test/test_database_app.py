from pathlib import Path

import pytest
from ase.db import connect

from asr.database.app import ASRDBApp
from asr.database.project import make_project
from asr.test.materials import Ag


@pytest.mark.ci
def test_simple_app(asr_tmpdir):
    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    app = ASRDBApp(tmpdir=tmpdir)
    database = connect("test_database.db")
    database.write(Ag)
    project = make_project(name="Test database", database=database)
    app.initialize_project(project)
