"""Database web application."""
import functools
import multiprocessing
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import flask.json as flask_json
from ase.db import connect
from ase.db.app import DBApp
from flask import Response, jsonify, redirect, render_template, send_file
from jinja2 import UndefinedError

import asr
import asr.database.browser as browser
from asr.core import ASRResult, UnknownDataFormat, decode_object

if TYPE_CHECKING:
    from asr.database.project import DatabaseProject
    from asr.database.key_descriptions import KeyDescriptions


class ASRDBApp(DBApp):
    def __init__(self, tmpdir, template_path: Path = Path(asr.__file__).parent.parent):
        self.tmpdir = tmpdir  # used to cache png-files
        super().__init__()

        self.flask.jinja_loader.searchpath.append(str(template_path))
        self.setup_app()
        self.setup_data_endpoints()

    def initialize_project(self, database, project: "DatabaseProject"):
        self.projects[project.name] = project.tospec(database)

    def setup_app(self):
        route = self.flask.route

        @route("/")
        def index():
            return render_template(
                "asr/database/templates/projects.html",
                projects=sorted(
                    [
                        (name, proj["title"], proj["database"].count())
                        for name, proj in self.projects.items()
                    ]
                ),
            )

        @route("/<project>/file/<uid>/<name>")
        def file(project, uid, name):
            assert project in self.projects
            path = self.tmpdir / f"{project}/{uid}-{name}"
            return send_file(str(path))

    def setup_data_endpoints(self):
        """Set endpoints for downloading data."""
        from ase.io.jsonio import MyEncoder

        self.flask.json_encoder = MyEncoder
        projects = self.projects

        route = self.flask.route

        @route("/<project_name>/row/<uid>/all_data")
        def get_all_data(project_name: str, uid: str):
            """Show details for one database row."""
            project = projects[project_name]
            uid_key = project["uid_key"]
            row = project["database"].get(
                "{uid_key}={uid}".format(uid_key=uid_key, uid=uid)
            )
            content = flask_json.dumps(row.data)
            return Response(
                content,
                mimetype="application/json",
                headers={"Content-Disposition": f"attachment;filename={uid}_data.json"},
            )

        @route("/<project_name>/row/<uid>/data")
        def show_row_data(project_name: str, uid: str):
            """Show details for one database row."""
            project = projects[project_name]
            uid_key = project["uid_key"]
            row = project["database"].get(
                "{uid_key}={uid}".format(uid_key=uid_key, uid=uid)
            )
            sorted_data = {
                key: value
                for key, value in sorted(row.data.items(), key=lambda x: x[0])
            }
            return render_template(
                "asr/database/templates/data.html",
                data=sorted_data,
                uid=uid,
                project_name=project_name,
            )

        @route("/<project_name>/row/<uid>/data/<filename>")
        def get_row_data_file(project_name: str, uid: str, filename: str):
            """Show details for one database row."""
            project = projects[project_name]
            uid_key = project["uid_key"]
            row = project["database"].get(
                "{uid_key}={uid}".format(uid_key=uid_key, uid=uid)
            )
            try:
                result = decode_object(row.data[filename])
                return render_template(
                    "asr/database/templates/result_object.html",
                    result=result,
                    filename=filename,
                    project_name=project_name,
                    uid=uid,
                )
            except (UnknownDataFormat, UndefinedError):
                return redirect(f"{filename}/json")

        @route("/<project_name>/row/<uid>/data/<filename>/json")
        def get_row_data_file_json(project_name: str, uid: str, filename: str):
            """Show details for one database row."""
            project = projects[project_name]
            uid_key = project["uid_key"]
            row = project["database"].get(
                "{uid_key}={uid}".format(uid_key=uid_key, uid=uid)
            )
            return jsonify(row.data.get(filename))


@contextmanager
def new_dbapp():
    with tempfile.TemporaryDirectory(prefix="asr-app-") as tmpdir:
        dbapp = ASRDBApp(Path(tmpdir))

        @dbapp.flask.template_filter()
        def asr_sort_key_descriptions(value):
            """Sort column drop down menu."""

            def sort_func(item):
                return item[1][1]

            return sorted(value.items(), key=sort_func)

        yield dbapp


def create_key_descriptions(db=None, extra_kvp_descriptions=None):
    from ase.db.web import create_key_descriptions

    from asr.core import read_json
    from asr.database.fromtree import parse_key_descriptions
    from asr.database.key_descriptions import key_descriptions

    flatten = {
        key: value
        for recipe, dct in key_descriptions.items()
        for key, value in dct.items()
    }

    if extra_kvp_descriptions is not None and Path(extra_kvp_descriptions).is_file():
        extras = read_json(extra_kvp_descriptions)
        flatten.update(extras)

    if db is not None:
        metadata = db.metadata
        if "keys" not in metadata:
            raise KeyError(
                "Missing list of keys for database. "
                "To fix this either: run database.fromtree again. "
                "or python -m asr.database.set_metadata DATABASEFILE."
            )
        keys = metadata.get("keys")
    else:
        keys = list(flatten)

    kd = {}
    for key in keys:
        description = flatten.get(key)
        if description is None:
            warnings.warn(f"Missing key description for {key}")
            continue
        kd[key] = description

    kd = {
        key: (desc["shortdesc"], desc["longdesc"], desc["units"])
        for key, desc in parse_key_descriptions(kd).items()
    }
    key_descriptions = make_key_descriptions_from_old_format(
        create_key_descriptions(kd)
    )
    return key_descriptions


def make_key_descriptions_from_old_format(
    key_descriptions: dict[str, tuple[str, str, str]]
) -> "KeyDescriptions":
    from asr.database.key_descriptions import KeyDescription, KeyDescriptions

    kd = {
        key: KeyDescription(short=value[0], long=value[1], unit=value[2])
        for key, value in key_descriptions.items()
    }
    return KeyDescriptions(kd)


def main(
    databases: List[str],
    host: str = "0.0.0.0",
    test: bool = False,
    extra_kvp_descriptions: Optional[str] = None,
) -> ASRResult:

    # The app uses threads, and we cannot call matplotlib multithreadedly.
    # Therefore we use a multiprocessing pool for the plotting.
    # We could use more cores, but they tend to fail to close
    # correctly on KeyboardInterrupt.
    pool = multiprocessing.Pool(1)
    with new_dbapp() as dbapp:
        try:
            _main(dbapp, databases, host, test, extra_kvp_descriptions, pool)
        finally:
            pool.close()
            pool.join()


def _main(dbapp, databases, host, test, extra_kvp_descriptions, pool):

    for database in databases:
        add_database_to_app(
            app=dbapp,
            dbfile=database,
            pool=pool,
            extra_kvp_descriptions=extra_kvp_descriptions,
        )

    flask = dbapp.flask

    if test:
        projects = dbapp.projects
        import traceback

        flask.testing = True
        with flask.test_client() as c:
            for name in projects:
                print(f"Testing {name}")
                c.get(f"/{name}/").data.decode()
                project = projects[name]
                db = project["database"]
                uid_key = project["uid_key"]
                n = len(db)
                uids = []
                for row in db.select(include_data=False):
                    uids.append(row.get(uid_key))
                    if len(uids) == n:
                        break
                print(len(uids))

                for i, uid in enumerate(uids):
                    url = f"/{name}/row/{uid}"
                    print(f"\rRows: {i + 1}/{len(uids)} {url}", end="", flush=True)
                    try:
                        c.get(url).data.decode()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        print()
                        row = db.get(uid=uid)
                        exc = traceback.format_exc()
                        exc += (
                            f"Problem with {uid}: "
                            f"Formula={row.formula} "
                            f"Crystal type={row.crystal_type}\n" + "-" * 20 + "\n"
                        )
                        with Path("errors.txt").open(mode="a") as fid:
                            fid.write(exc)
                            print(exc)
    else:
        flask.run(host=host, debug=True)


def add_database_to_app(
    app: ASRDBApp,
    dbfile: str,
    pool: Optional[multiprocessing.Pool] = None,
    extra_kvp_descriptions: str = None,
):
    tmpdir = get_app_tmpdir(app)
    make_project_tempdir(dbfile, tmpdir)
    name = get_project_name(dbfile)
    database = connect_to_database(dbfile)
    project = make_database_project(
        name, pool, tmpdir, database, extra_kvp_descriptions
    )
    app.initialize_project(database, project)


def get_app_tmpdir(app):
    tmpdir = app.tmpdir
    return tmpdir


def make_database_project(name, pool, tmpdir, db, extra_kvp_descriptions):
    from asr.database.project import make_project_from_database_metadata

    row_to_dict_function = make_row_to_dict_function(pool, tmpdir)
    key_descriptions = create_key_descriptions(db, extra_kvp_descriptions)
    project = make_project_from_database_metadata(
        db.metadata,
        key_descriptions=key_descriptions,
        name=name,
        row_to_dict_function=row_to_dict_function,
    )
    return project


def connect_to_database(dbfile):
    db = connect(dbfile, serial=True)
    return db


def make_project_tempdir(dbfile, tmpdir):
    name = get_project_name(dbfile)
    (tmpdir / name).mkdir()


def get_project_name(dbfile):
    name = Path(dbfile).name
    return name


def make_row_to_dict_function(pool, tmpdir):
    """Make a special row_to_dict function.

    Udd´ses a pool for figure creation and places figures in tmpdir.
    """

    def layout(*args, **kwargs):
        return browser.layout(*args, pool=pool, **kwargs)

    row_to_dict_function = functools.partial(
        browser.row_to_dict,
        layout_function=layout,
        tmpdir=tmpdir,
    )
    return row_to_dict_function
