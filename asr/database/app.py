"""Database web application."""
from typing import List
import multiprocessing
import tempfile
from pathlib import Path
import warnings
from contextlib import contextmanager

from flask import render_template, send_file, Response, jsonify, redirect
import flask.json as flask_json
from jinja2 import UndefinedError
from ase.db import connect
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.geometry import cell_to_cellpar
from ase.formula import Formula
from ase.db.app import DBApp

import asr
from asr.core import ASRResult, decode_object, UnknownDataFormat


class ASRDBApp(DBApp):
    def __init__(self, tmpdir, template_path=None):
        self.tmpdir = tmpdir  # used to cache png-files
        super().__init__()

        if template_path is None:
            template_path = Path(asr.__file__).parent.parent
        self.flask.jinja_loader.searchpath.append(  # pylint: disable=no-member
            str(template_path)
        )

        self.setup_app()
        self.setup_data_endpoints()

    def initialize_project(self, database, extra_kvp_descriptions=None, pool=None):
        row_to_dict_function = self.make_row_to_dict_function(pool)
        project = get_project_from_database(extra_kvp_descriptions, database)
        spec = project.tospec()
        spec["row_to_dict_function"] = row_to_dict_function
        self.projects[project.name] = spec
        (self.tmpdir / project.name).mkdir()

    def make_row_to_dict_function(self, pool):
        from asr.database import browser
        from functools import partial
        from asr.database.project import row_to_dict

        def layout(*args, **kwargs):
            return browser.layout(*args, pool=pool, **kwargs)

        row_to_dict_function = partial(
            row_to_dict,
            layout_function=layout,
            tmpdir=self.tmpdir,
        )
        return row_to_dict_function

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
    from asr.database.key_descriptions import key_descriptions
    from asr.database.fromtree import parse_key_descriptions
    from ase.db.web import create_key_descriptions

    flatten = {
        key: value
        for recipe, dct in key_descriptions.items()
        for key, value in dct.items()
    }

    if extra_kvp_descriptions is not None:
        flatten.update(extra_kvp_descriptions)

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

    return create_key_descriptions(kd)


def get_project_from_database(extra_kvp_descriptions, database):
    from asr.core import read_json

    if extra_kvp_descriptions is not None and Path(extra_kvp_descriptions).is_file():
        extras = read_json(extra_kvp_descriptions)
    else:
        extras = None

    db = connect(database, serial=True)
    metadata = db.metadata
    name = metadata.get("name", Path(database).name)

    key_descriptions = create_key_descriptions(db, extras)
    title = metadata.get("title", name)
    uid_key = metadata.get("uid", "uid")
    default_columns = metadata.get("default_columns", ["formula", "uid"])
    table_template = str(
        metadata.get(
            "table_template",
            "asr/database/templates/table.html",
        )
    )
    search_template = str(
        metadata.get("search_template", "asr/database/templates/search.html")
    )
    row_template = str(metadata.get("row_template", "asr/database/templates/row.html"))

    from asr.database.project import DatabaseProject

    project = DatabaseProject(
        name=name,
        title=title,
        key_descriptions=key_descriptions,
        uid_key=uid_key,
        database=db,
        default_columns=default_columns,
        table_template=table_template,
        search_template=search_template,
        row_template=row_template,
    )
    return project


class Summary:
    def __init__(self, row, key_descriptions, create_layout, prefix=""):
        self.row = row

        atoms = Atoms(cell=row.cell, pbc=row.pbc)
        self.size = kptdensity2monkhorstpack(atoms, kptdensity=1.8, even=False)

        self.cell = [["{:.3f}".format(a) for a in axis] for axis in row.cell]
        par = ["{:.3f}".format(x) for x in cell_to_cellpar(row.cell)]
        self.lengths = par[:3]
        self.angles = par[3:]

        stress = row.get("stress")
        if stress is not None:
            stress = ", ".join("{0:.3f}".format(s) for s in stress)
        self.stress = stress

        self.formula = Formula(row.formula).convert("metal").format("html")

        kd = key_descriptions
        self.layout = create_layout(row, kd, prefix)

        dipole = row.get("dipole")
        if dipole is not None:
            dipole = ", ".join("{0:.3f}".format(d) for d in dipole)
        self.dipole = dipole

        data = row.get("data")
        if data:
            data = ", ".join(data)
        self.data = data
        constraints = row.get("constraints")
        if constraints:
            constraints = ", ".join(c.__class__.__name__ for c in self.constraints)
        self.constraints = constraints


def main(
    databases: List[str],
    host: str = "0.0.0.0",
    test: bool = False,
    extra_kvp_descriptions: str = "key_descriptions.json",
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
    projects = dbapp.projects
    for database in databases:
        dbapp.initialize_project(database, extra_kvp_descriptions, pool)

    flask = dbapp.flask

    if test:
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
