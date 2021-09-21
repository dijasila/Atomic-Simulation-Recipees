from pathlib import Path

import asr
from ase.db.app import app
from asr.database.app import row_to_dict, create_key_descriptions
# from asr.database.app import setup_data_endpoints
from asr.database.browser import layout

__all__ = ['row_to_dict', 'create_key_descriptions', 'layout']

path = Path(asr.__file__).parent.parent
app.jinja_loader.searchpath.append(str(path))  # noqa

title = 'Computational 1D materials database'

default_columns = ['formula', 'is_magnetic',
                   'hform', 'gap', 'crystal_type']

uid_key = 'uid'


def handle_query(args):
    parts = []
    if args['dyn_phonons'] != 'all':
        parts.append('dynamic_stability_phonons=' + args['dyn_phonons'])
    if args['dyn_stiffness'] != 'all':
        parts.append('dynamic_stability_stiffness=' + args['dyn_stiffness'])
    if args['from_tdyn'] > '1':
        parts.append('thermodynamic_stability_level>=' + args['from_tdyn'])
    if args['to_tdyn'] < '3':
        parts.append('thermodynamic_stability_level<=' + args['to_tdyn'])
    if args['from_gap']:
        parts.append(args['xc'] + '>=' + args['from_gap'])
    if args['to_gap']:
        parts.append(args['xc'] + '<=' + args['to_gap'])
    if args['is_magnetic']:
        parts.append('is_magnetic=' + args['is_magnetic'])
    # We only want to present the first class materials to users
    parts.append('first_class_material=True')
    return ','.join(parts)


def connect_endpoints(app, proj):
    """Set endpoints for downloading data."""
    pass  # setup_data_endpoints()
