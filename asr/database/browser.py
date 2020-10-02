from asr.core import command, option, dct_to_result, ASRResult
import copy
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import traceback
import os

import matplotlib.pyplot as plt
from ase.db.row import AtomsRow
from ase.db.core import float_to_time_string, now

assert sys.version_info >= (3, 4)

plotlyjs = (
    '<script src="https://cdn.plot.ly/plotly-latest.min.js">' + '</script>')
external_libraries = [plotlyjs]

unique_key = 'uid'

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'savefig.dpi': 200}
plt.rcParams.update(**params)


def create_table(row,  # AtomsRow
                 header,  # List[str]
                 keys,  # List[str]
                 key_descriptions,  # Dict[str, Tuple[str, str, str]]
                 digits=3  # int
                 ):  # -> Dict[str, Any]
    """Create table-dict from row."""
    table = []
    for key in keys:
        if key == 'age':
            age = float_to_time_string(now() - row.ctime, True)
            table.append(('Age', age))
            continue
        value = row.get(key)
        if value is not None:
            if isinstance(value, float):
                value = '{:.{}f}'.format(value, digits)
            elif not isinstance(value, str):
                value = str(value)
            desc, unit = key_descriptions.get(key, ['', key, ''])[1:]
            if unit:
                value += ' ' + unit
            table.append((desc, value))
    return {'type': 'table',
            'header': header,
            'rows': table}


def miscellaneous_section(row, key_descriptions, exclude):
    """Make help function for adding a "miscellaneous" section.

    Create table with all keys except those in exclude.
    """
    misckeys = (set(key_descriptions)
                | set(row.key_value_pairs)) - set(exclude)
    misc = create_table(row, ['Items', ''], sorted(misckeys), key_descriptions)
    return ('Miscellaneous', [[misc]])


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


def merge_panels(page):
    """Merge panels which have the same title.

    Also merge tables with same first entry in header.
    """
    # Update panels
    for title, panels in page.items():
        panels = sorted(panels, key=lambda x: x['sort'])

        panel = {'title': title,
                 'columns': [[], []],
                 'plot_descriptions': [],
                 'sort': panels[0]['sort']}
        known_tables = {}
        for tmppanel in panels:
            for column in tmppanel['columns']:
                for ii, item in enumerate(column):
                    if isinstance(item, dict):
                        if item['type'] == 'table':
                            if 'header' not in item:
                                continue
                            header = item['header'][0]
                            if header in known_tables:
                                known_tables[header]['rows']. \
                                    extend(item['rows'])
                                column[ii] = None
                            else:
                                known_tables[header] = item

            columns = tmppanel['columns']
            if len(columns) == 1:
                columns.append([])

            columns[0] = [item for item in columns[0] if item]
            columns[1] = [item for item in columns[1] if item]
            panel['columns'][0].extend(columns[0])
            panel['columns'][1].extend(columns[1])
            panel['plot_descriptions'].extend(tmppanel['plot_descriptions'])
        page[title] = panel


def extract_recipe_from_filename(filename: str):
    """Parse filename and return recipe name."""
    pattern = re.compile('results-(.*)\.json')  # noqa
    m = pattern.match(filename)
    return m.group(1)


def is_results_file(filename):
    return filename.startswith('results-') and filename.endswith('.json')


class RowWrapper:

    def __init__(self, row):
        self._row = row
        self._data = copy.deepcopy(row.data)

    def __getattr__(self, key):
        """Wrap attribute lookup of AtomsRow."""
        if key == 'data':
            return self._data
        return getattr(self._row, key)

    def __contains__(self, key):
        """Wrap contains of atomsrow."""
        return self._row.__contains__(key)


def layout(row: AtomsRow,
           key_descriptions: Dict[str, Tuple[str, str, str]],
           prefix: Path) -> List[Tuple[str, List[List[Dict[str, Any]]]]]:
    """Page layout."""
    page = {}
    exclude = set()

    row = RowWrapper(row)
    for key, value in row.data.items():
        if is_results_file(key):
            obj = dct_to_result(value)
        else:
            obj = key
        row.data[key] = obj
        assert row.data[key] == obj

    # Locate all webpanels
    for filename in row.data:
        if not is_results_file(filename):
            continue
        result = row.data[filename]
        panels = result.format_as('ase_webpanel', row, key_descriptions)
        if not panels:
            continue

        for thispanel in panels:
            assert 'title' in thispanel, f'No title in {filename} webpanel'
            panel = {'columns': [[], []],
                     'plot_descriptions': [],
                     'sort': 99}
            panel.update(thispanel)
            paneltitle = panel['title']
            if paneltitle in page:
                page[paneltitle].append(panel)
            else:
                page[paneltitle] = [panel]

    merge_panels(page)
    page = [panel for _, panel in page.items()]
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
                try:
                    function(row, *(str(path) for path in paths))
                except Exception:
                    if os.environ.get('ASRTESTENV', False):
                        raise
                    else:
                        traceback.print_exc()
                plt.close('all')
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


@command('asr.database.browser')
@option('--database', type=str)
@option('--only-figures', is_flag=True,
        help='Dont show browser, just save figures')
def main(database: str = 'database.db',
         only_figures: bool = False) -> ASRResult:
    """Open results in web browser."""
    import subprocess
    from pathlib import Path

    custom = Path(__file__)

    cmd = f'python3 -m ase db {database} -w -M {custom}'
    if only_figures:
        cmd += ' -l'
    print(cmd)
    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit(1)


if __name__ == '__main__':
    main.cli()
