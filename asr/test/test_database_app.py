from pathlib import Path

import pytest
from ase.db import connect

from asr.database.app import ASRDBApp
from asr.database.project import make_project
from asr.test.materials import Ag


@pytest.fixture
def project(asr_tmpdir):
    database = connect("test_database.db")
    database.write(Ag)
    project = make_project(name="database.db", database=database, key_descriptions={})
    return project


@pytest.fixture
def client(project):
    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    app = ASRDBApp(tmpdir=tmpdir)
    app.initialize_project(project)
    app.flask.testing = True
    with app.flask.test_client() as client:
        yield client


@pytest.mark.ci
def test_single_project_home_page(client):
    response = client.get("/").data
    assert b"<a href='/database.db/'>database.db</a>" in response


@pytest.mark.ci
def test_single_project_project_home_page(client):
    response = client.get("/database.db/").data.decode()
    assert "<h1>database.db</h1>" in response
