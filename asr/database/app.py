from asr.core import command, option, argument

import tempfile
from pathlib import Path

from ase.db import connect
from ase.db.app import app, projects
from flask import render_template, send_file

tmpdir = Path(tempfile.mkdtemp(prefix='asr-app-'))  # used to cache png-files
path = Path(__file__).parent
app.jinja_loader.searchpath.append(str(path / 'templates'))


@app.route('/')
def index():
    return render_template(
        'projects.html',
        projects=sorted([(name,
                          proj['title'],
                          proj['database'].count())
                         for name, proj in projects.items()]))


@app.route('/<project>/file/<uid>/<name>')
def file(project, uid, name):
    assert project in projects
    path = tmpdir / f'{project}-{uid}-{name}'  # XXXXXXXXXXX
    return send_file(str(path))


def handle_query(args):
    return args['query']


def initialize_project(database):
    from asr.database import browser
    from ase.db.web import create_key_descriptions
    db = connect(database)
    metadata = db.metadata
    name = metadata.get('name', database)

    projects[name] = {
        'name': name,
        'title': metadata.get('title', name),
        'uid_key': metadata.get('uid', 'uid'),
        'key_descriptions': create_key_descriptions(
            metadata['key_descriptions']),
        'database': db,
        'handle_query_function': handle_query,
        'row_to_dict_function': browser.row_to_dict,
        'default_columns': metadata.get('default_columns'),
        'search_template': str(metadata.get('search_template', 'search.html')),
        'row_template': str(metadata.get('row_template', 'row.html'))
    }


@command()
@argument('database')
@option('--host', help='Host address.')
def main(database, host='0.0.0.0'):
    initialize_project(database)
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main.cli()
