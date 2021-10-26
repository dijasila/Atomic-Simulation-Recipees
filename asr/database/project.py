"""Define an object that represents a database project."""
import multiprocessing.pool
import pathlib
import runpy
import typing
from dataclasses import dataclass, field

from ase.db.core import Database
from asr.database.browser import layout

if typing.TYPE_CHECKING:
    from asr.database.key_descriptions import KeyDescriptions


def args2query(args):
    return args["query"]


def row_to_dict(row, project):
    from asr.database.app import Summary

    def create_layout(*args, **kwargs):
        return project.layout_function(*args, pool=project.pool, **kwargs)

    project_name = project["name"]
    uid = row.get(project["uid_key"])
    s = Summary(
        row,
        create_layout=create_layout,
        key_descriptions=project["key_descriptions"],
        prefix=str(project.tmpdir / f"{project_name}/{uid}-"),
    )
    return s


@dataclass
class DatabaseProject:
    """Class that represents a database project.

    Parameters
    ----------
    name
        The name of the database project.
    title
        The title of the database object.
    database
        A database connection
    key_descriptions
        Key descriptions used by the web application
    uid_key
        Key to be used as unique identifier
    row_to_dict_function
        A function that takes (row, project) as input and produces an
        object (normally a dict) that is handed to the row template
        also specified in this project.
    handle_query_function
        A function that takes a query tuple and returns a query tuple.
        Useful for doing translations when the query uses aliases
        for values, for example to convert stability=low to stability=1.
    default_columns
        Default columns that the application should show on the search page.
    table_template
        Path to the table jinja-template.
        The table template shows the rows of the database.
    row_template
        Path to the row jinja-template.
        The row template is responsible for showing a detailed description
        of a particular row.
    search_template
        Path to the search jinja-template. The search template embeds the table
        template and is responsible for formatting the search field.
    """

    name: str
    title: str
    database: Database
    key_descriptions: "KeyDescriptions"
    uid_key: str = "uid"
    tmpdir: typing.Optional[pathlib.Path] = None
    row_to_dict_function: typing.Callable = row_to_dict
    handle_query_function: typing.Callable = args2query
    default_columns: typing.List[str] = field(
        default_factory=lambda: list(["formula", "id"])
    )
    table_template: str = "asr/database/templates/table.html"
    row_template: str = "asr/database/templates/row.html"
    search_template: str = "asr/database/templates/search.html"
    layout_function: typing.Callable = layout
    pool: typing.Optional[multiprocessing.pool.Pool] = None
    template_search_path: typing.Optional[str] = None

    def __getitem__(self, item):
        return self.__dict__[item]


def make_project(
    name: str,
    database: Database,
    title: typing.Optional[str] = None,
    key_descriptions: typing.Optional["KeyDescriptions"] = None,
    uid_key: str = "uid",
    tmpdir: typing.Optional[pathlib.Path] = None,
    row_to_dict_function: typing.Callable = row_to_dict,
    handle_query_function: typing.Callable = args2query,
    default_columns: typing.Optional[typing.List[str]] = None,
    table_template: str = "asr/database/templates/table.html",
    row_template: str = "asr/database/templates/row.html",
    search_template: str = "asr/database/templates/search.html",
):
    """Make database project.

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
        The key to use as UID, by default "id"
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
        The table template for the project,
        by default "asr/database/templates/table.html"
    row_template : str, optional
        The row template for the project,
        by default "asr/database/templates/row.html"
    search_template : str, optional
        The search template for the project,
        by default "asr/database/templates/search.html"
    """
    if title is None:
        title = name

    if key_descriptions is None:
        from asr.database.app import create_default_key_descriptions
        key_descriptions = create_default_key_descriptions()

    if default_columns is None:
        default_columns = ["formula", "id"]

    return DatabaseProject(
        name=name,
        title=title,
        database=database,
        tmpdir=tmpdir,
        key_descriptions=key_descriptions,
        uid_key=uid_key,
        handle_query_function=handle_query_function,
        row_to_dict_function=row_to_dict_function,
        default_columns=default_columns,
        table_template=table_template,
        search_template=search_template,
        row_template=row_template,
    )


def make_project_from_pyfile(path: str) -> DatabaseProject:
    """Make a database project from a Python file.

    Parameters
    ----------
    path : str
        Path to a Python file that defines some or all of
        the attributes that defines a database project, e.g.
        name=, title=. At a minimum name and database needs
        to be defined.

    Returns
    -------
    DatabaseProject
        A database project constructed from the attributes defined
        in the python file.
    """
    module = runpy.run_path(str(path))
    return make_project_from_dict(module)


def make_project_from_dict(dct):
    values = {}
    keys = set(
        (
            "name",
            "title",
            "database",
            "key_descriptions",
            "uid_key",
            "handle_query_function",
            "row_to_dict_function",
            "default_columns",
            "table_template",
            "search_template",
            "row_template",
        )
    )

    for key in keys:
        if key in dct:
            values[key] = dct[key]
    return make_project(**values)
