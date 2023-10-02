"""Database web application."""
from typing import List
import multiprocessing
import tempfile
from pathlib import Path

from flask import render_template, send_file, Response, jsonify, redirect
from flask.json.provider import JSONProvider
import flask.json
from jinja2 import UndefinedError
from ase.db import connect
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.geometry import cell_to_cellpar
from ase.formula import Formula
from ase.db.app import new_app
from ase.db.project import DatabaseProject
from ase.io.jsonio import encode as ase_encode

import asr
from asr.core import (command, option, argument, ASRResult,
                      decode_object, UnknownDataFormat)


def create_key_descriptions():
    from asr.database.key_descriptions import key_descriptions
    from ase.db.core import get_key_descriptions as get_ase_keydescs

    all_keydescs_flat = dict(get_ase_keydescs())

    # We should check for clashes here.
    for recipe, dct in key_descriptions.items():
        all_keydescs_flat.update(dct)

    # Should warn if descriptions are None or empty string
    return all_keydescs_flat


class Summary:
    def __init__(self, row, key_descriptions, prefix=''):
        self.row = row

        atoms = Atoms(cell=row.cell, pbc=row.pbc)
        self.size = kptdensity2monkhorstpack(atoms,
                                             kptdensity=1.8,
                                             even=False)

        self.cell = [['{:.3f}'.format(a) for a in axis] for axis in row.cell]
        par = ['{:.3f}'.format(x) for x in cell_to_cellpar(row.cell)]
        self.lengths = par[:3]
        self.angles = par[3:]

        self.stress = row.get('stress')
        if self.stress is not None:
            self.stress = ', '.join('{0:.3f}'.format(s) for s in self.stress)

        self.formula = Formula(row.formula).convert('metal').format('html')

        from asr.database.browser import layout
        layout, subpanel = layout(row, key_descriptions, prefix)
        self.layout = layout
        self.subpanel = subpanel

        self.dipole = row.get('dipole')
        if self.dipole is not None:
            self.dipole = ', '.join('{0:.3f}'.format(d) for d in self.dipole)

        self.data = row.get('data')
        if self.data:
            self.data = ', '.join(self.data.keys())

        self.constraints = row.get('constraints')
        if self.constraints:
            self.constraints = ', '.join(c.__class__.__name__
                                         for c in self.constraints)


class WebApp:
    def __init__(self, app, projects, tmpdir):
        self.app = app
        self.tmpdir = tmpdir
        self.projects = projects

    def initialize_project(self, database, pool=None):
        from asr.database import browser

        db = connect(database, serial=True)
        metadata = db.metadata
        name = metadata.get("name", Path(database).name)

        tmpdir = self.tmpdir
        # Make temporary directory
        (tmpdir / name).mkdir()

        def layout(*args, **kwargs):
            return browser.layout(*args, pool=pool, **kwargs)

        metadata = db.metadata

        # much duplication of initialization
        project = ASRProject(
            name=name,
            title=metadata.get("title", name),
            key_descriptions=create_key_descriptions(),
            database=db,
            tempdir=tmpdir,
            uid_key=metadata.get("uid", "uid"),
            default_columns=metadata.get("default_columns",
                                         ["formula", "uid"]))

        self.projects[name] = project


def setup_app(route_slash=True, tmpdir=None):
    # used to cache png-files:
    tmpdir = tmpdir or Path(tempfile.mkdtemp(prefix="asr-app-"))

    path = Path(asr.__file__).parent.parent
    projects = {}
    app = new_app(projects)
    app.jinja_loader.searchpath.append(str(path))

    if route_slash:
        @app.route("/")
        def index():
            return render_template(
                "asr/database/templates/projects.html",
                projects=sorted([
                    (name, project.title, project.database.count())
                    for name, project in projects.items()
                ]))

    @app.route("/<project>/file/<uid>/<name>")
    def file(project, uid, name):
        assert project in projects
        path = tmpdir / f"{project}/{uid}-{name}"  # XXXXXXXXXXX
        return send_file(str(path))

    @app.template_filter()
    def sort_key_descriptions(value):
        """Sort column drop down menu."""
        def sort_func(item):
            # These items are ('id', <KeyDescription>)
            # We (evidently) sort by longdesc.
            return item[1].longdesc

        return sorted(value.items(), key=sort_func)

    webapp = WebApp(app, projects, tmpdir)
    setup_data_endpoints(webapp)
    return webapp


class _ASEJsonProvider(JSONProvider):
    def dumps(self, obj):
        return ase_encode(obj)


def setup_data_endpoints(webapp):
    """Set endpoints for downloading data."""

    projects = webapp.projects
    app = webapp.app
    app.json = _ASEJsonProvider(app)

    @app.route('/<project_name>/row/<uid>/all_data')
    def get_all_data(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        row = project.uid_to_row(uid)
        content = flask.json.dumps(row.data)
        return Response(
            content,
            mimetype='application/json',
            headers={'Content-Disposition':
                     f'attachment;filename={uid}_data.json'})

    @app.route('/<project_name>/row/<uid>/data')
    def show_row_data(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        row = project.uid_to_row(uid)
        sorted_data = {key: value for key, value
                       in sorted(row.data.items(), key=lambda x: x[0])}
        return render_template(
            'asr/database/templates/data.html',
            data=sorted_data, uid=uid, project_name=project_name)

    @app.route('/<project_name>/row/<uid>/data/<filename>')
    def get_row_data_file(project_name: str, uid: str, filename: str):
        """Show details for one database row."""
        project = projects[project_name]
        row = project.uid_to_row(uid)
        try:
            result = decode_object(row.data[filename])
            return render_template(
                'asr/database/templates/result_object.html',
                result=result,
                filename=filename,
                project_name=project_name,
                uid=uid,
            )
        except (UnknownDataFormat, UndefinedError):
            return redirect(f'{filename}/json')

    @app.route('/<project_name>/row/<uid>/data/<filename>/json')
    def get_row_data_file_json(project_name: str, uid: str, filename: str):
        """Show details for one database row."""
        project = projects[project_name]
        row = project.uid_to_row(uid)
        return jsonify(row.data.get(filename))


class ASRProject(DatabaseProject):
    _asr_templates = Path('asr/database/templates/')

    def __init__(self, *, uid_key, tempdir, **kwargs):
        self.tempdir = tempdir
        super().__init__(**kwargs)
        self.uid_key = uid_key

    def row_to_dict(self, row):
        project_name = self.name
        uid = row.get(self.uid_key)
        s = Summary(row,
                    key_descriptions=self.key_descriptions,
                    prefix=str(self.tempdir / f'{project_name}/{uid}-'))
        return s

    # XXX copypasty
    def get_table_template(self):
        return self._asr_templates / 'table.html'

    def get_search_template(self):
        return self._asr_templates / 'search.html'

    def get_row_template(self):
        return self._asr_templates / 'row.html'


@command()
@argument("databases", nargs=-1, type=str)
@option("--host", help="Host address.", type=str)
@option("--test", is_flag=True, help="Test the app.")
def main(databases: List[str], host: str = "0.0.0.0",
         test: bool = False) -> ASRResult:

    # The app uses threads, and we cannot call matplotlib multithreadedly.
    # Therefore we use a multiprocessing pool for the plotting.
    # We could use more cores, but they tend to fail to close
    # correctly on KeyboardInterrupt.
    pool = multiprocessing.Pool(1)
    try:
        _main(databases, host, test, pool)
    finally:
        pool.close()
        pool.join()


def _main(databases, host, test, pool):
    webapp = setup_app()
    projects = webapp.projects
    app = webapp.app

    for database in databases:
        webapp.initialize_project(database, pool)

    if test:
        app.testing = True
        from asr.database.app_testing import run_testing
        run_testing(app, projects)
    else:
        webapp.app.run(host=host, debug=True)


if __name__ == "__main__":
    main.cli()
