from pathlib import Path
from ase.db import connect
from asr.database import DatabaseProject, App

project = DatabaseProject.from_pyfile("project.py")
other_project = DatabaseProject(
    name="Some other project",
    title="Title of the other project",
    database=connect("database.db"),
)
app = App()
app.add_project(project)
app.add_project(other_project)
app.run()
