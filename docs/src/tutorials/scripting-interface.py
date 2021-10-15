from ase.db import connect
from asr.database import make_project_from_pyfile, make_project, App

project = make_project_from_pyfile("project.py")
other_project = make_project(
    name="Some other project",
    database=connect("database.db")
)
app = App()
app.initialize_project(project)
app.initialize_project(other_project)
app.run()
