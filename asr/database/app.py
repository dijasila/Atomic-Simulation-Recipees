from asr import command, option, argument

import importlib
import tempfile
from functools import partial
from pathlib import Path

from ase.db import connect
from ase.db.app import app, projects, handle_query
from ase.db.web import create_key_descriptions
from flask import render_template, send_file

from cmr import project_names
from cmr.web import row_to_dict


tmpdir = Path(tempfile.mkdtemp(prefix='asr-app-'))  # used to cache png-files
path = Path(__file__).parent
app.jinja_loader.searchpath.append(str(path))


@app.route('/')
def index():
    return render_template(
        'templates/projects.html',
        projects=sorted([(name,
                          proj['title'],
                          proj['database'].count())
                         for name, proj in projects.items()]))


@app.route('/<project>/file/<uid>/<name>')
def file(project, uid, name):
    assert project in projects
    path = tmpdir / f'{project}-{uid}-{name}'  # XXXXXXXXXXX
    return send_file(str(path))


def initialize_project(database):
    from asr.database import browser
    db = connect(database)
    metadata = db.metadata
    assert 'name' in metadata, \
        'You must set a database name in database metadata'
    search = path / 'templates' / 'search.html'
    row = path / 'templates' / 'row.html'

    name = metadata['name']

    projects[name] = {
        'name': name,
        'uid_key': metadata.get('uid', 'uid'),
        'key_descriptions': metadata['key_descriptions'],
        'database': db,
        'row_to_dict_function': browser.row_to_dict,
        'default_columns': metadata.get('default_columns'),
        'search_template': metadata.get('search', search),
        'row_template': metadata.get('search', row)
    }


@command()
@argument('database', help='Database file.')
def main(database, host='0.0.0.0'):
    initialize_project(database)
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main.cli()
