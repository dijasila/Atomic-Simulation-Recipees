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
    name = Path(database).with_suffix('')
    print(name)

    proj = {'name': name}

    search = path / name / 'search.html'
    if search.is_file():
        proj['search_template'] = f'{name}/search.html'
    else:
        proj['search_template'] = 'templates/search.html'

    row = path / name / 'row.html'
    if row.is_file():
        proj['row_template'] = f'{name}/row.html'
    else:
        proj['row_template'] = 'templates/row.html'

    db = connect(path / f'../docs/{name}/{name}.db')
    proj['database'] = db

    mod = importlib.import_module(f'cmr.{name}.custom')
    proj['default_columns'] = mod.default_columns
    proj['title'] = mod.title
    proj['uid_key'] = getattr(mod, 'uid_key', 'id')
    proj['key_descriptions'] = create_key_descriptions(mod.key_descriptions)

    if hasattr(mod, 'row_to_dict'):
        proj['row_to_dict_function'] = mod.row_to_dict
    else:
        proj['row_to_dict_function'] = partial(
            row_to_dict,
            layout_function=getattr(mod, 'layout', None),
            tmpdir=tmpdir)

    proj['handle_query_function'] = getattr(mod, 'handle_query',
                                            handle_query)

    if hasattr(mod, 'connect_endpoints'):
        mod.connect_endpoints(app, proj)

    projects[name] = proj


@command()
@argument('database', help='Database file.')
def main(database, host='0.0.0.0'):
    initialize_project(database)
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main.cli()
