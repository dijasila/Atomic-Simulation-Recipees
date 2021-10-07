from asr.database.key_descriptions import KeyDescription, KeyDescriptions
from asr.database.project import make_project_description
from asr.database.app import connect_to_database, make_asr_app, add_database_to_app
from asr.database.key_descriptions import make_key_descriptions


def test_make_key_description():
    kd = KeyDescriptions(
        natoms=KeyDescription(long="Number of atoms", short="natoms", unit="Number")
    )
    kd2 = make_key_descriptions(natoms=("natoms", "Number of atoms", "Number"))


def test_simple_database_project():
    description = make_project_description(
        name="Test database",
        key_descriptions=dict(natoms=("natoms", "Number of atoms", "Number")),
    )
    database = connect_to_database("database.db")
    app = make_asr_app()
    app.add_project(database, description)
    add_database_to_app(app, dbfile)
