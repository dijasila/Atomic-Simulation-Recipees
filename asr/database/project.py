"""Define an object that represents a database project.

"""
import pathlib
import typing
import dataclass
from asr.database.key_descriptions import KeyDescriptions


@dataclass
class DatabaseProject:
    """Class that represents a database project.

    Contains
    """

    name: str
    title: str
    key_descriptions: KeyDescriptions
    uid_key: str
    database: ase.db.Database
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


def make_project_from_database_metadata(
    database: str,
    extra_kvp_descriptions: typing.Optional[dict] = None,
) -> DatabaseProject:
    db = connect(database, serial=True)
    metadata = db.metadata
    name = metadata.get("name", Path(database).name)

    (self.tmpdir / name).mkdir()

    def layout(*args, **kwargs):
        return browser.layout(*args, pool=pool, **kwargs)

    metadata = db.metadata
    return {
        "name": name,
        "title": metadata.get("title", name),
        "key_descriptions": create_key_descriptions(db, extra_kvp_descriptions),
        "uid_key": metadata.get("uid", "uid"),
        "database": db,
        "handle_query_function": args2query,
        "row_to_dict_function": partial(
            row_to_dict,
            layout_function=layout,
            tmpdir=self.tmpdir,
        ),
        "default_columns": metadata.get("default_columns", ["formula", "uid"]),
        "table_template": str(
            metadata.get(
                "table_template",
                "asr/database/templates/table.html",
            )
        ),
        "search_template": str(
            metadata.get("search_template", "asr/database/templates/search.html")
        ),
        "row_template": str(
            metadata.get("row_template", "asr/database/templates/row.html")
        ),
    }


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
