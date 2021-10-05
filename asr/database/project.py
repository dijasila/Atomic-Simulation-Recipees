"""Define an object that represents a database project.

"""

class DatabaseProject:
    """Class that represents a database projects.
    
    Contains 
    """

    def group_rows(self):
        ...        


def make_project_from_configuration_file(path: pathlib.Path):
    """Construct a database from a configarion file."""
    config = read_configuration_file(path)
    project = make_project_from_config(config)
    return project


def read_configuration_file(path:)