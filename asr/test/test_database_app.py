from pathlib import Path

import pytest
from ase.db import connect

from asr.database.app import App
from asr.database.project import make_project
from asr.test.materials import Ag


@pytest.fixture
def database_with_one_row(asr_tmpdir):
    database = connect("test_database.db")
    database.write(Ag)
    return database


@pytest.fixture
def project(database_with_one_row):
    project = make_project(
        name="database.db", database=database_with_one_row, key_descriptions={}
    )
    return project


@pytest.fixture
def client(project):
    tmpdir = Path("tmp/")
    tmpdir.mkdir()
    app = App(tmpdir=tmpdir)
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


def test_add_extra_kvp_descriptions(project):
    from asr.database.app import add_extra_kvp_descriptions

    key_name = "some_key_name"
    description = "Some description."
    extras = {key_name: description}

    add_extra_kvp_descriptions([project], extras=extras)

    assert key_name in project.key_descriptions
    assert project.key_descriptions[key_name] == description


def test_setting_custom_row_to_dict_function(project):
    from asr.database.app import set_custom_row_to_dict_function

    prev = project.row_to_dict_function
    set_custom_row_to_dict_function(project, Path("tmpdir"), pool=None)
    new = project.row_to_dict_function
    assert prev is not new


def test_app_running(project, mocker):
    from asr.database.app import run_app, App

    # app.run blocks, so we patch it to check the other logic of the function.
    mocker.patch.object(App, "run")
    run_app(host="0.0.0.0", test=False, projects=[project], extras={})
