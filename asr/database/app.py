from asr.core import command, option, argument

import tempfile
from pathlib import Path

from ase.db import connect
from ase.db.app import app, projects
from flask import render_template, send_file
import asr
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.geometry import cell_to_cellpar
from ase.utils import formula_metal

tmpdir = Path(tempfile.mkdtemp(prefix="asr-app-"))  # used to cache png-files

path = Path(asr.__file__).parent.parent
app.jinja_loader.searchpath.append(str(path))


class Summary:
    def __init__(self, row, key_descriptions, create_layout,
                 subscript=None, prefix=''):
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

        self.formula = formula_metal(row.numbers)
        if subscript:
            self.formula = subscript.sub(r'<sub>\1</sub>', self.formula)

        kd = key_descriptions
        self.layout = create_layout(row, kd, prefix)

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


def setup_app():
    @app.route("/")
    def index():
        return render_template(
            "asr/database/templates/projects.html",
            projects=sorted(
                [
                    (name, proj["title"], proj["database"].count())
                    for name, proj in projects.items()
                ]
            ),
        )

    @app.route("/<project>/file/<uid>/<name>")
    def file(project, uid, name):
        assert project in projects
        path = tmpdir / f"{project}-{uid}-{name}"  # XXXXXXXXXXX
        return send_file(str(path))


def handle_query(args):
    return args["query"]


def row_to_dict(row, project, layout_function, tmpdir):
    from asr.database.browser import layout
    project_name = project['name']
    uid = row.get(project['uid_key'])
    s = Summary(row,
                create_layout=layout,
                key_descriptions=project['key_descriptions'],
                prefix=str(tmpdir / f'{project_name}-{uid}-'))
    return s


def initialize_project(database):
    from asr.database import browser
    from ase.db.web import create_key_descriptions
    from functools import partial

    db = connect(database)
    metadata = db.metadata
    name = metadata.get("name", database)

    projects[name] = {
        "name": name,
        "title": metadata.get("title", name),
        "key_descriptions": create_key_descriptions(
            metadata["key_descriptions"]
        ),
        "uid_key": metadata.get("uid", "uid"),
        "database": db,
        "handle_query_function": handle_query,
        "row_to_dict_function": partial(
            row_to_dict, layout_function=browser.layout, tmpdir=tmpdir
        ),
        "default_columns": metadata.get("default_columns"),
        "search_template": str(
            metadata.get(
                "search_template", "asr/database/templates/search.html"
            )
        ),
        "row_template": str(
            metadata.get("row_template", "asr/database/templates/row.html")
        ),
    }


@command()
@argument("databases", nargs=-1)
@option("--host", help="Host address.")
def main(databases, host="0.0.0.0"):
    for database in databases:
        initialize_project(database)
    setup_app()
    app.run(host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main.cli()
