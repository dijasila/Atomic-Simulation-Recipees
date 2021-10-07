"""Define an object that represents a database project."""
import pathlib
import typing
from dataclasses import dataclass, field

from asr.database.browser import args2query, row_to_dict

if typing.TYPE_CHECKING:
    from asr.database.key_descriptions import KeyDescriptions


@dataclass
class DatabaseProject:
    """Class that represents a database project."""

    name: str
    title: str
    key_descriptions: "KeyDescriptions"
    uid_key: str = "uid"
    row_to_dict_function: callable = row_to_dict
    tmpdir: typing.Optional[pathlib.Path] = None
    handle_query_function: callable = args2query
    default_columns: list[str] = field(default_factory=lambda: list(["formula", "uid"]))
    table_template: str = "asr/database/templates/table.html"
    row_template: str = "asr/database/templates/row.html"
    search_template: str = "asr/database/templates/search.html"

    def tospec(self, database) -> dict:
        """Compile dict spec for "database" useful for ASE web application."""
        spec = {
            "database": database,
            **self.__dict__,
        }
        return spec


def make_project_from_database_metadata(
    metadata: dict,
    key_descriptions: typing.Optional[dict] = None,
    name: typing.Optional[str] = None,
    row_to_dict_function: typing.Optional[callable] = None,
    handle_query_function: typing.Optional[callable] = None,
) -> DatabaseProject:

    name = name or metadata.get("name")
    table_template = (
        str(
            metadata.get(
                "table_template",
                "asr/database/templates/table.html",
            )
        ),
    )
    search_template = str(
        metadata.get("search_template", "asr/database/templates/search.html")
    )
    row_template = str(metadata.get("row_template", "asr/database/templates/row.html"))
    default_columns = metadata.get("default_columns", ["formula", "uid"])
    return DatabaseProject(
        name=name,
        title=metadata.get("title", name),
        key_descriptions=key_descriptions,
        uid_key=metadata.get("uid", "uid"),
        handle_query_function=handle_query_function,
        row_to_dict_function=row_to_dict_function,
        default_columns=default_columns,
        table_template=table_template,
        search_template=search_template,
        row_template=row_template,
    )


def make_project_description(
    name: str,
    title: typing.Optional[str] = None,
    key_descriptions: typing.Optional["KeyDescriptions"] = None,
    uid_key: str = "uid",
    row_to_dict_function: callable = row_to_dict,
    tmpdir: typing.Optional[pathlib.Path] = None,
    handle_query_function: callable = args2query,
    default_columns: list[str] = field(
        default_factory=lambda: list(["formula", "uid"])
    ),
    table_template: str = "asr/database/templates/table.html",
    row_template: str = "asr/database/templates/row.html",
    search_template: str = "asr/database/templates/search.html",
):
    """Make project description.

    Used as input to the ASR app to give it information about key descriptions etc.

    Parameters
    ----------
    name : str
        The name of the project
    title : str, optional
        The title of the database, defaults to name
    key_descriptions : KeyDescriptions
        Descriptions for key value pairs.
    uid_key : str, optional
        The key to use as UID, by default "uid"
    row_to_dict_function : callable, optional
        A function that takes a row and returns a dict, by default row_to_dict
    tmpdir : typing.Optional[pathlib.Path], optional
        The temporary directory associated with the database app, by default None
    handle_query_function : callable, optional
        A functional that turns turn the app args into a database query,
        by default args2query
    default_columns : list[str], optional
        Default project columns, by default ["formula", "uid"]
    table_template : str, optional
        The table template for the project, by default "asr/database/templates/table.html"
    row_template : str, optional
        The row template for the project, by default "asr/database/templates/row.html"
    search_template : str, optional
        The search template for the project, by default "asr/database/templates/search.html"
    """

    from asr.database.key_descriptions import make_key_descriptions

    if not title:
        title = name

    key_descriptions = make_key_descriptions(key_descriptions)

    return DatabaseProject(
        name=name,
        title=title,
        key_descriptions=key_descriptions,
        uid_key=uid_key,
        handle_query_function=handle_query_function,
        row_to_dict_function=row_to_dict_function,
        default_columns=default_columns,
        table_template=table_template,
        search_template=search_template,
        row_template=row_template,
    )
