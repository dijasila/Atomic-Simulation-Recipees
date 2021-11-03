import pathlib

import pytest

from asr.database.project import DatabaseProject


@pytest.fixture
def project(database_with_one_row):

    dct = dict(
        name="project.name",
        title="project.title",
        database=database_with_one_row,
    )

    pjt = DatabaseProject(**dct)

    return pjt


@pytest.mark.ci
def test_project_from_namespace_has_name(project):
    assert project.name == "project.name"


@pytest.mark.ci
def test_project_from_namespace_has_database(project, database_with_one_row):
    assert project.database == database_with_one_row


@pytest.mark.ci
def test_project_from_namespace_has_title(project):
    assert project.title == "project.title"
    assert project.name == "project.name"


@pytest.mark.ci
def test_project_from_pyfile(asr_tmpdir):
    txt = """
from ase.db import connect
name = "name_of_database"
title = "Title of database"
database = connect("dbname.db")
key_descriptions = dict(
    key_name=("short", "long", "unit"),
)
"""
    filename = "project.py"
    pathlib.Path(filename).write_text(txt)
    pjt = DatabaseProject.from_pyfile(filename)
    assert pjt.name == "name_of_database"
    assert pjt.title == "Title of database"


@pytest.mark.ci
def test_project_from_database(asr_tmpdir):
    from ase.db import connect
    filename = 'database.db'
    db = connect(filename)
    name = "Name of database"
    db.metadata = dict(
        name=name,
        keys=["formula"],
    )
    pjt = DatabaseProject.from_database(filename)
    assert pjt.name == name
    assert pjt.title == name
