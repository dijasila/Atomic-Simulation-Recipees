from pathlib import Path
import pytest
from ase.db import connect

from asr.database.app import App
from asr.database.project import DatabaseProject
from asr.test.materials import Ag


@pytest.fixture
def database_with_one_row(asr_tmpdir):
    database = connect("test_database.db")
    database.write(Ag)
    database.metadata = dict(keys=[])
    return database


@pytest.fixture
def project(database_with_one_row):
    project = DatabaseProject(
        name="database.db",
        title="database.db",
        database=database_with_one_row,
        uid_key="formula",
        tmpdir=Path("tmpdir/"),
    )

    return project


@pytest.fixture
def client(project):
    app = App()
    app.add_project(project)
    app.flask.testing = True
    with app.flask.test_client() as client:
        yield client


@pytest.mark.ci
def test_single_project_home_page(client, project):
    response = client.get("/").data.decode()
    assert f"<a href='/database.db/'>{project.name}</a>" in response


@pytest.mark.ci
def test_single_project_database_home_page(client, project):
    response = client.get("/database.db/").data.decode()
    assert f"<h1>{project.name}</h1>" in response
    # XXX We cannot test the table view because the DOMContentLoaded
    # XXX event doesn't fire for some reason.
    # assert "Displaying rows" in response


@pytest.mark.ci
def test_single_project_material_page(client):
    response = client.get("/database.db/row/Ag")
    assert response.status_code == 200


@pytest.mark.ci
def test_add_extra_kvp_descriptions(project):
    from asr.database.app import add_extra_kvp_descriptions

    key_name = "some_key_name"
    description = "Some description."
    extras = {key_name: description}

    add_extra_kvp_descriptions([project], extras=extras)

    assert key_name in project.key_descriptions
    assert project.key_descriptions[key_name] == description


@pytest.mark.ci
def test_app_running(project, mocker):
    from asr.database.app import run_app, App

    # app.run blocks, so we patch it to check the other logic of the function.
    mocker.patch.object(App, "run")
    run_app(host="0.0.0.0", test=False, projects=[project])
