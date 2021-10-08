import pytest
from asr.database.project import get_project_from_namespace
from types import SimpleNamespace


@pytest.mark.ci
def test_make_project_from_module(database_with_one_row):

    namespace = SimpleNamespace(
        name="project.name",
        database=database_with_one_row,
    )

    project = get_project_from_namespace(namespace)

    assert project.name == "project.name"