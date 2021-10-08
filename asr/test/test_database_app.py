from pathlib import Path

import pytest
from ase.db import connect

from asr.database.app import ASRDBApp
from asr.database.project import make_project
from asr.test.materials import Ag


@pytest.mark.ci
def test_simple_app(asr_tmpdir):
    database = connect("test_database.db")
    database.write(Ag)

    project = make_project(name="database.db", database=database, key_descriptions={})

    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    app = ASRDBApp(tmpdir=tmpdir)
    app.initialize_project(project)

    app.flask.testing = True
    with app.flask.test_client() as client:
        response = client.get("/database.db/").data.decode()
        assert "<h1>database.db</h1>" in response
