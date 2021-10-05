"""Define an object that represents a database project.

"""
import pathlib


@dataclass
class DatabaseProject:
    """Class that represents a database project.

    Contains
    """

    name: str
    title: str
    key_descriptions: KeyDescriptions
    uid_key: str
    database: pathlib.Path
    row_to_dict_function: callable
    handle_query_function: callable = args2query
    default_columns: list[str] = ["formula", "uid"]
    table_template: str = "asr/database/templates/table.html"
    row_template: str = "asr/database/templates/row.html"

    def todict(self) -> dict:
        """Compile dict spec useful for ASE web application."""

        return self.__dict__

    def group_rows(self):
        ...


def args2query(args):
    return args["query"]


def row_to_dict(row, project, layout_function, tmpdir):
    project_name = project["name"]
    uid = row.get(project["uid_key"])
    s = Summary(
        row,
        create_layout=layout_function,
        key_descriptions=project["key_descriptions"],
        prefix=str(tmpdir / f"{project_name}/{uid}-"),
    )
    return s


class DatabaseView:
    """Class that represents a database view."""

    def key_descriptions(self):
        ...


class Config:
    pass


def make_project_from_configuration_file(path: pathlib.Path):
    """Construct a database from a configarion file."""
    config = read_configuration_file(path)
    project = make_project_from_config(config)
    return project


def read_configuration_file(path: pathlib.Path) -> Config:
    ...


def make_project_from_config():
    return DatabaseProject()
