"""Database web application."""
import multiprocessing
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Any

import flask.json as flask_json
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.db.app import Database, DBApp
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import MyEncoder
from flask import Response, jsonify, redirect, render_template, send_file, abort
from jinja2 import UndefinedError

import asr
from asr.core import UnknownDataFormat, decode_object

from asr.database.project import DatabaseProject


class App(DBApp):
    """App that can browse multiple database projects."""

    def __init__(self):
        super().__init__()

        self.flask.jinja_loader.searchpath.append(  # pylint: disable=no-member
            Path(asr.__file__).parent.parent
        )

        self._setup_app()
        self._setup_data_endpoints()

    def run(self, host="localhost", debug=False):
        """Run app.

        Parameters
        ----------
        host : str, optional
            The host address, for example "0.0.0.0", by default "localhost"
        debug : bool, optional
            Run server in debug mode, by default False

        """
        self.initialize()
        self.flask.run(host=host, debug=debug)

    def add_project(self, project: DatabaseProject):
        """Add a project to application.

        Parameters
        ----------
        project : DatabaseProject
            The project to be added.
        """
        self.projects[project.name] = project

    def add_projects(self, projects: List[DatabaseProject]):
        """Add multiple projects to application.

        Parameters
        ----------
        projects : List[DatabaseProject]
            Databases to be added.
        """
        for project in projects:
            self.add_project(project)

    def initialize(self):
        for project in self.projects.values():
            if project.tmpdir is None:
                continue

            if not project.tmpdir.exists():
                project.tmpdir.mkdir()

            actual_tmpdir = project.tmpdir / project.name
            actual_tmpdir.mkdir(exist_ok=True, parents=True)

            if project.template_search_path is not None:
                self.flask.jinja_loader.searchpath.append(  # pylint: disable=no-member
                    project.template_search_path
                )

    def _setup_app(self):
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
            if project.tmpdir is None:
                abort(Response("Project has no tmpdir, cannot locate file."))
            path = project.tmpdir / f"{project}/{uid}-{name}"
            return send_file(str(path))

        @self.flask.template_filter()
        def asr_sort_key_descriptions(value):
            """Sort column drop down menu."""

            def sort_func(item):
                return item[1][1]

            return sorted(value.items(), key=sort_func)

        @self.flask.template_filter()
        def make_help_button(value: Any):
            print(value, type(value))
            if hasattr(value, "description"):
                return value.description
            return ""

        @self.flask.template_filter()
        def debug(text):
            print(text)
            return ""

    def _setup_data_endpoints(self):
        """Set endpoints for downloading data."""
        self.flask.json_encoder = MyEncoder
        projects = self.projects

        route = self.flask.route

        @route("/<project_name>/row/<uid>/all_data")
        def get_all_data(project_name: str, uid: str):
            """Show details for one database row."""
            project = projects[project_name]
            uid_key = project["uid_key"]
            row = project["database"].get(f"{uid_key}={uid}")
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
            row = project["database"].get(f"{uid_key}={uid}")
            sorted_data = dict(sorted(row.data.items(), key=lambda x: x[0]))
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
            row = project["database"].get(f"{uid_key}={uid}")
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
            row = project["database"].get(f"{uid_key}={uid}")
            return jsonify(row.data.get(filename))


def create_default_key_descriptions(db: Database = None):
    from asr.database.key_descriptions import key_descriptions

    flatten = {
        key: value
        for recipe, dct in key_descriptions.items()
        for key, value in dct.items()
    }

    if db is not None:
        keys = get_db_keys(db)
        flatten = pick_subset_of_keys(keys, flatten)

    return convert_to_ase_compatible_key_descriptions(flatten)


def get_db_keys(db):
    metadata = db.metadata
    if "keys" not in metadata:
        raise KeyError(
            "Missing list of keys for database. "
            "To fix this either: run database.fromtree again. "
            "or python -m asr.database.set_metadata DATABASEFILE."
        )
    keys = metadata.get("keys")
    return keys


def convert_to_ase_compatible_key_descriptions(key_descriptions):
    from ase.db.web import create_key_descriptions

    from asr.database.fromtree import parse_key_descriptions

    kd = {
        key: (desc["shortdesc"], desc["longdesc"], desc["units"])
        for key, desc in parse_key_descriptions(key_descriptions).items()
    }

    return create_key_descriptions(kd)


def pick_subset_of_keys(keys, key_descriptions):
    kd = {}
    for key in keys:
        description = key_descriptions.get(key)
        if description is None:
            warnings.warn(f"Missing key description for {key}")
            continue
        kd[key] = description
    return kd


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
            constraints = ", ".join(c.__class__.__name__ for c in constraints)
        self.constraints = constraints


def add_extra_kvp_descriptions(projects, extras):
    """Update existing project key descriptions with extras."""
    for project in projects:
        project.key_descriptions.update(extras)


def main(
    filenames: List[str],
    host: str = "0.0.0.0",
    test: bool = False,
    extra_kvp_descriptions_file: str = "key_descriptions.json",
):
    """Start database app.

    Start the database based on filenames. Assigns a pool to all projects
    that doesn't already have a pool. Also automatically assigns a tmpdir
    to all projects that doesn't already have this. If you don't want this
    behaviour consider using :func:`asr.database.app.run_app` directly.

    Parameters
    ----------
    filenames : List[str]
        List of databases (.db) or project configuration files (.py).
    host : str, optional
        Host address, by default "0.0.0.0"
    test : bool, optional
        Whether to query all rows of all input projects/databases, by default False
    extra_kvp_descriptions_file : str, optional
        File containing extra key descriptions for the database,
        by default "key_descriptions.json"
    """
    pool = multiprocessing.Pool(1)
    projects = convert_files_to_projects(filenames, pool=pool)
    extras = get_key_descriptions_from_file(extra_kvp_descriptions_file)

    with tempfile.TemporaryDirectory(prefix='asr-app-') as tmpdir:
        # For some reason MyPy complains about giving a string argument
        # to the Path object. Don't know why. Ignoring.
        tmpdir = Path(tmpdir)  # type: ignore
        set_tmpdir_on_projects_if_missing(tmpdir, projects)
        try:
            run_app(projects, extras, host, test)
        finally:
            pool.close()
            pool.join()


def set_tmpdir_on_projects_if_missing(tmpdir, projects):
    for project in projects:
        if project.tmpdir is None:
            project.tmpdir = tmpdir


def set_pool_on_projects_if_missing(pool, projects: List["DatabaseProject"]):
    for project in projects:
        if project.pool is None:
            project.pool = pool


def run_app(
    projects: List["DatabaseProject"],
    extra_key_descriptions: Optional[dict] = None,
    host: str = "0.0.0.0",
    test: bool = False,
):
    """Start and run a database application.

    Convencience function for creating an Application object,
    initializing projects and running the application. The
    application can be viewed in a browser at "host" address.

    Parameters
    ----------
    projects : List[DatabaseProject]
        Add these projects to the application
    extra_key_descriptions : Optional[dict]
        Add these key value pair descriptions to the projects before
        starting the application, by default None
    host : str, optional
        The host address, by default "0.0.0.0"
    test : bool, optional
        Whether to query all rows of all input projects/databases, by default False
    """
    if extra_key_descriptions:
        add_extra_kvp_descriptions(projects, extra_key_descriptions)

    dbapp = App()
    dbapp.add_projects(projects)
    if test:
        check_rows_of_all_projects(dbapp)
    else:
        dbapp.run(host=host, debug=True)


def convert_files_to_projects(filenames, pool=None):
    projects = []
    for filename in filenames:
        if filename.endswith("py"):
            project = DatabaseProject.from_pyfile(filename)
        elif filename.endswith("db"):
            project = DatabaseProject.from_database(filename)
        else:
            raise ValueError
        projects.append(project)
    set_pool_on_projects_if_missing(pool, projects)
    return projects


def get_key_descriptions_from_file(extra_kvp_descriptions_file):
    from asr.core import read_json

    if (
        extra_kvp_descriptions_file is not None
        and Path(extra_kvp_descriptions_file).is_file()
    ):
        extras = read_json(extra_kvp_descriptions_file)
    else:
        extras = {}
    return extras


def check_rows_of_all_projects(dbapp):
    import traceback

    flask = dbapp.flask
    projects = dbapp.projects

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
