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
    from asr.utils import get_recipes
    page = []
    exclude = set()

    # Locate all webpanels
    recipes = get_recipes()
    for recipe in recipes:
        if not recipe.webpanel:
            continue
        if not recipe.done:
            continue
        panels = recipe.webpanel(row, key_descriptions)
        page.extend(panels)

    # Sort sections if they have a sort key
    page = [x for x in sorted(page, key=lambda x: x.get('sort', 99))]

    misc_title, misc_columns = miscellaneous_section(row, key_descriptions,
                                                     exclude)
    misc_panel = {'title': misc_title,
                  'columns': misc_columns}
    page.append(misc_panel)

    # Get descriptions of figures that are created by all webpanels
    plot_descriptions = []
    for panel in page:
        plot_descriptions.extend(panel.get('plot_descriptions', []))

    # List of functions and the figures they create:
    missing = set()  # missing figures
    for desc in plot_descriptions:
        function = desc['function']
        filenames = desc['filenames']
        paths = [Path(prefix + filename) for filename in filenames]
        for path in paths:
            if not path.is_file():
                # Create figure(s) only once:
                function(row, *(str(path) for path in paths))
                for path in paths:
                    if not path.is_file():
                        path.write_text('')  # mark as missing
                break
        for path in paths:
            if path.stat().st_size == 0:
                missing.add(path)

    # We convert the page into ASE format
    asepage = []
    for panel in page:
        asepage.append((panel['title'], panel['columns']))

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
    for title, columns in asepage:
        columns = [[block for block in column if ok(block)]
                   for column in columns]
        if any(columns):
            final_page.append((title, columns))

    return final_page
