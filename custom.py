import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt
from ase.db.row import AtomsRow
from ase.db.summary import create_table, miscellaneous_section
assert sys.version_info >= (3, 4)

title = 'Computational materials database'

plotlyjs = (
    '<script src="https://cdn.plot.ly/plotly-latest.min.js">' + '</script>')
external_libraries = [
    plotlyjs,
]

unique_key = 'uid'

default_columns = [
    'formula', 'prototype', 'magstate', 'class', 'spacegroup', 'hform', 'gap',
    'work_function'
]

stabilities = {1: 'low', 2: 'medium', 3: 'high'}

special_keys = [('SELECT', 'prototype'), ('SELECT', 'class'),
                ('SRANGE', 'dynamic_stability_level', stabilities),
                ('SRANGE', 'thermodynamic_stability_level', stabilities),
                ('SELECT', 'magstate'),
                ('RANGE', 'gap', 'Band gap range [eV]',
                 [('PBE', 'gap'), ('G0W0@PBE', 'gap_gw'),
                  ('GLLBSC', 'gap_gllbsc'), ('HSE@PBE', 'gap_hse')])]

params = {
    'legend.fontsize': 'large',
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    'savefig.dpi': 200
}
plt.rcParams.update(**params)


def val2str(row, key: str, digits=2) -> str:
    value = row.get(key)
    if value is not None:
        if isinstance(value, float):
            value = '{:.{}f}'.format(value, digits)
        elif not isinstance(value, str):
            value = str(value)
    else:
        value = ''
    return value


def fig(filename: str, link: str = None) -> 'Dict[str, Any]':
    """Shortcut for figure dict."""
    dct = {'type': 'figure', 'filename': filename}
    if link:
        dct['link'] = link
    return dct


def table(row, title, keys, kd={}, digits=2):
    return create_table(row, [title, 'Value'], keys, kd, digits)


def layout(row: AtomsRow, key_descriptions: 'Dict[str, Tuple[str, str, str]]',
           prefix: str) -> 'List[Tuple[str, List[List[Dict[str, Any]]]]]':
    """Page layout."""
    import asr
    page = []
    things = []
    exclude = set()

    # Locate all webpanels
    from importlib import import_module
    pathlist = Path(asr.__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name

        module = import_module('asr.' + name)

        try:
            print('module', name)
            panel, newthings = module.webpanel(row, key_descriptions)
            if panel:
                page.append(panel)
            if newthings:
                things.extend(newthings)
        except AttributeError:
            continue

    page += [miscellaneous_section(row, key_descriptions, exclude)]

    # List of functions and the figures they create:
    missing = set()  # missing figures
    for func, filenames in things:
        paths = [Path(prefix + filename) for filename in filenames]
        for path in paths:
            if not path.is_file():
                # Create figure(s) only once:
                func(row, *(str(path) for path in paths))
                for path in paths:
                    if not path.is_file():
                        path.write_text('')  # mark as missing
                break
        for path in paths:
            if path.stat().st_size == 0:
                missing.add(path)

    def ok(block):
        if block is None:
            return False
        if block['type'] == 'table':
            return block['rows']
        if block['type'] != 'figure':
            return True
        if Path(prefix + block['filename']) in missing:
            return False
        return True

    # Remove missing figures from layout:
    final_page = []
    for title, columns in page:
        columns = [[block for block in column if ok(block)]
                   for column in columns]
        if any(columns):
            final_page.append((title, columns))

    return final_page


group = 'Utility'
