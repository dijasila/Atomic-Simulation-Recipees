import pytest
from asr.database.project import get_project_from_namespace
from types import SimpleNamespace


@pytest.fixture
def project(database_with_one_row):

    namespace = SimpleNamespace(
        name="project.name",
        database=database_with_one_row,
    )

    project = get_project_from_namespace(namespace)

    return project


@pytest.mark.ci
def test_project_from_namespace_has_name(project):
    assert project.name == "project.name"


@pytest.mark.ci
def test_project_from_namespace_has_database(project, database_with_one_row):
    assert project.database == database_with_one_row


@pytest.mark.ci
def test_project_from_namespace_has_title(project):
    assert project.title == "project.name"
    assert project.name == "project.name"
