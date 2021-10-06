from asr.database.project import DatabaseProject
from asr.database.key_descriptions import KeyDescription, KeyDescriptions


def test_simple_project():
    key_descriptions = KeyDescriptions(
        natoms=KeyDescription(long="Number of atoms", short="natoms", unit="Number")
    )

    project = DatabaseProject(
        name="Test database",
        title="Title of test database",
        key_descriptions=key_descriptions,
    )

    assert project.name == "Test database"
