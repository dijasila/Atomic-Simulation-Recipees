from asr.core import (command, option, decode_object,
                      ASRResult, get_recipe_from_name)
import copy
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import traceback
import os
from .webpanel import WebPanel
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from ase.db.row import AtomsRow
from ase.db.core import float_to_time_string, now
from asr.webpages.appresources import HTMLStringFormat

assert sys.version_info >= (3, 4)

plotlyjs = (
    '<script src="https://cdn.plot.ly/plotly-latest.min.js">' + '</script>')
external_libraries = [plotlyjs]


def create_table(row: AtomsRow, header: List[str], keys: List[str],
                 key_descriptions: Dict[str, Tuple[str, str, str]] = {},
                 digits: int = 3): #-> Dict[str, Any]
    """
    Assemble information in a dictionary from AtomsRow object to create an
    HTML table.

    :params row: An AtomsRow or ASRResults object to create the table from
    :params header: The table header for each column in the table
    :params keys: The AtomsRow key which you want to include in the table
    :params key_descriptions:
    :params digits: The significant figures to round a float in the table
    :return: {'type': 'table', 'header': [header for table], 'rows': [data for
    each row in the table]}
    """
    table = []
    for key in keys:
        if key == 'age':
            age = float_to_time_string(now() - row.ctime, True)
            table.append(('Age', age))
            continue
        value = row.get(key)
        if value is not None:
            if isinstance(value, float):
                old_value = value
                value = '{:.{}f}'.format(value, digits)
                if hasattr(old_value, '__explanation__'):
                    value = describe_entry(value, old_value.__explanation__)
            elif not isinstance(value, str):
                value = str(value)

            from ase.db.project import KeyDescription
            nokeydesc = KeyDescription(key, key, '', '')  # XXX
            keydesc = key_descriptions.get(key, nokeydesc)
            unit = keydesc.unit
            desc = keydesc.shortdesc  # XXX shortdesc or longdesc?
            # This code has the tuple interchanged!!

            # longdesc, desc, unit = key_descriptions.get(key, ['', key, ''])
            if hasattr(key, '__explanation__'):
                desc = describe_entry(desc, key.__explanation__)
            if unit:
                value += ' ' + unit
            table.append([desc, value])

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
    return ('Miscellaneous Stuff', [[misc]])


def link_section(row, key_descriptions, exclude):
    """
    Make help function for adding a "links" section.
    Create links table in the links panel.
    Create table with all keys except those in exclude.
    (exclude is not used this is nonsense)
    """
    try:
        links = row.data['links']
    except KeyError:
        return ('Links', [[]])

    link_table = create_table(row=row, header=['Links', 'Value'], keys=[],
                              key_descriptions=key_descriptions, digits=2)

    for link in links:
        linkname = HTMLStringFormat.href(link[0], link[1])
        linktype = link[2]
        link_table['rows'].extend([[linkname, linktype]])

    return ('Links', [[link_table]])


value_type_to_explained_type = {}


def describe_entry(value, description, title='Information'):
    """Describe website entry.

    This function sets an __explanation__ attribute on the given object
    which is used by the web application to generate additional explanations.
    """
    description = HTMLStringFormat.normalize_string(description)
    if hasattr(value, '__explanation__'):
        if value.__explanation__ == '':
            value.__explanation__ += description
        else:
            value.__explanation__ += '\n' + description
        value.__explanation_title__ = bold(title)
        return value

    value_type = type(value)
    if value_type in value_type_to_explained_type:
        value = value_type_to_explained_type[value_type](value)
        value.__explanation__ = description
        value.__explanation_title__ = bold(title)
        return value

    class ExplainedType(value_type):

        __explanation__: str
        __explanation_title__: str

    value_type_to_explained_type[value_type] = ExplainedType
    return describe_entry(value, description, title)


def describe_entries(rows, description):
    for ir, row in enumerate(rows):
        for ic, value in enumerate(row):
            if isinstance(value, dict):
                raise ValueError(f'Incompatible value={value}')
            value = describe_entry(value, description)
            rows[ir][ic] = value
    return rows


def dict_to_list(dct, indent=0, char=' ', exclude_keys: set = set()):
    lst = []
    for key, value in dct.items():
        if key in exclude_keys:
            continue
        if value is None:
            continue
        if isinstance(value, dict):
            lst2 = dict_to_list(value,
                                indent=indent + 2,
                                char=char,
                                exclude_keys=exclude_keys)
            lst.extend([indent * char + f'<b>{key}</b>='] + lst2)
        else:
            lst.append(indent * char + f'<b>{key}</b>={value}')
    return lst

div = HTMLStringFormat.div()
li = HTMLStringFormat.lst()
bold = HTMLStringFormat.bold()
dt = HTMLStringFormat.dt()
dd = HTMLStringFormat.dd()
par = HTMLStringFormat.par()
pre = HTMLStringFormat('pre').wrap_txt_with_tag()
code = HTMLStringFormat('code').wrap_txt_with_tag()
br = '<br>'

static_article_links = {'C2DB': HTMLStringFormat.href(
    """S. Haastrup et al. The Computational 2D Materials Database: high-throughput
modeling and discovery of atomically thin crystals, 2D Mater. 5 042002
(2018).""",
    'https://doi.org/10.1088/2053-1583/aacfc1'
)
}


def make_panel_description(text, articles=None):

    if articles:
        articles = (
            bold('Relevant articles:')
            + HTMLStringFormat.indent_lst([
                static_article_links.get(article, article) for article in articles]
            )
        )
        elements = [text, articles]
    else:
        elements = [text]

    return HTMLStringFormat('<br>').combine_elements(items=elements)


def entry_parameter_description(data, name, exclude_keys: set = set()):
    """Make a parameter description.

    Parameters
    ----------
    data: dict
        Data object containing result objects (typically row.data).
    name: str
        Name of recipe from which to extract parameters, e.g. "asr.gs@calculate".
    exclude_keys: set
        Set of keys to exclude from parameter description.

    """
    recipe = get_recipe_from_name(name)
    link_name = HTMLStringFormat('<a>').get_recipe_href(asr_name=name)

    if (f'results-{name}.json' in data
            and 'params' in getattr(data[f'results-{name}.json'],
                                    'metadata', {})):
        metadata = data[f'results-{name}.json'].metadata
        params = metadata.params
        # header = ''
        # asr_name = (metadata.asr_name if 'asr_name' in metadata
        #             else name)  # Fall back to name as best guess for asr_name
        # link_name = HTMLStringFormat.get_recipe_href(asr_name, name=name)
    else:
        params = recipe.get_defaults()
        # header = ('No parameters can be found, meaning that '
        #           'the recipe was probably run with the '
        #           'default parameter shown below\n'
        #           '<b>Default:</b>')
        # link_name = HTMLStringFormat.get_recipe_href(name)

    lst = dict_to_list(params, exclude_keys=exclude_keys)
    string = pre(code('\n'.join(lst)))
    description = (
        bold(f'Parameters ({link_name})')
        + br
        + string
    )

    return description


def fig(filename: str, link: str = None,
        caption: str = None) -> 'Dict[str, Any]':
    """Shortcut for figure dict."""
    dct = {'type': 'figure', 'filename': filename}
    if link:
        dct['link'] = link
    if caption:
        dct['caption'] = caption
    return dct


def matrixtable(M, digits=2, unit='',
                rowlabels=None, columnlabels=None, title=None):
    shape_of_M = np.shape(M)
    shape = (shape_of_M[0] + 1, shape_of_M[1] + 1)

    rows = []
    for i in range(0, shape[0]):
        rows.append([])
        for j in range(0, shape[1]):
            rows[i].append("")

    for column_index in range(shape[1]):
        if column_index == 0 and title is not None:
            rows[0][0] = bold(title)
        elif column_index > 0 and columnlabels is not None:
            rows[0][column_index] = bold(columnlabels[column_index - 1])

    for row_index in range(shape[0]):
        if row_index > 0:
            rows[row_index][0] = bold(rowlabels[row_index - 1])

    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            value = M[i - 1][j - 1]
            if digits is None:
                rows[i][j] = value
                if unit != '':
                    raise TypeError(
                        f"input unit ({unit}) can't be set because digits "
                        "is None! When setting 'unit' please specify 'digits' "
                        "as well.")
            else:
                rows[i][j] = '{:.{}f}{}'.format(value, digits, unit)

    table = dict(type='table',
                 rows=rows)
    return table


# Example predefined grouping structure as a dictionary
grouped_strings = {
    'Band structures': ['Electronic band structure (HSE06@PBE)',
                        'Electronic band structure (PBE)',
                        'Electronic band structure (G0W0)'],
    'Polarizability': ['Optical polarizability', 'Infrared polarizability'],
    'Magnetic properties': ['Basic magnetic properties (PBE)',
                            'Basic magnetic properties (PBE+U)']
}

def merge_panels(page):
    """
    Merge panels which have the same title.
    Also merge tables with same first entry in header.
    Set the subpanel flag if the panel title is contained with the sub-panel
    headers
    """
    # Update panels
    for title, panels in page.items():
        panels = sorted(panels, key=lambda x: x['sort'])

        panel = {'title': title,
                 'columns': [[], []],
                 'plot_descriptions': [],
                 'sort': panels[0]['sort']}
        known_tables = {}
        # create the normal panels for the template
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

        # check if there are subpanel to create in the list of webpanels
        # The band structures definitely need subpanels (PBE, HSE06, G0W0).
        # Now we will soon also have PBE+U for band structure and magnetic
        # properties (only for materials including any of the 3d magnetic
        # metals), so the Magnetic properties will also have a subpanel (PBE
        # and PBE+U)
        def group_strings(string, grouped_strings):
            """
            when given a string and a dictionary of lists, the function
            searches the list and returns the key of the list in which the
            string is contained within
            """
            for key, values in grouped_strings.items():
                if any(val in string for val in values):
                    return key
            return None

        # check if the panel name is in the sub panel group
        panel_is_subpanel = group_strings(panel['title'], grouped_strings)
        if panel_is_subpanel:
            panel['subpanel'] = panel_is_subpanel
        panel = WebPanel(**panel)
        page[title] = panel


def extract_recipe_from_filename(filename: str):
    """Parse filename and return recipe name."""
    pattern = re.compile(r'results-(.*)\.json')  # noqa
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

    def __getitem__(self, key):
        """Get key directly."""
        return self._row[key]

    def __getstate__(self):
        """Help pickle overcome the troubles due to __getattr__.

        We need to provide getstate/setstate to prevent recursion error
        when unpickling.
        """
        return vars(self)

    def __setstate__(self, dct):
        """See __getstate__."""
        self.__dict__.update(dct)

    def __contains__(self, key):
        """Wrap contains of atomsrow."""
        return self._row.__contains__(key)


def parse_row_data(data: dict):
    newdata = {}
    for key, value in data.items():
        if is_results_file(key):
            obj = decode_object(value)

            # Below is to support old C2DB databases that contain
            # hacked result files with no asr_name
            if not isinstance(obj, ASRResult):
                recipename = extract_recipe_from_filename(key)
                value['__asr_hacked__'] = recipename
                obj = decode_object(value)
        else:
            obj = value
        newdata[key] = obj
    return newdata


def runplot_clean(plotfunction, *args):
    plt.close('all')
    value = plotfunction(*args)
    plt.close('all')
    return value


def generate_plots(row, prefix, plot_descriptions, pool):
    missing = set()
    for desc in plot_descriptions:
        function = desc['function']
        filenames = desc['filenames']
        paths = [Path(prefix + filename) for filename in filenames]
        for path in paths:
            if not path.is_file():
                # Create figure(s) only once:
                strpaths = [str(path) for path in paths]
                try:
                    args = [function, row] + strpaths
                    if pool is None:
                        runplot_clean(*args)
                    else:
                        pool.apply(runplot_clean, args)
                except Exception:
                    if os.environ.get('ASRTESTENV', False):
                        raise
                    else:
                        traceback.print_exc()

                for path in paths:
                    if not path.is_file():
                        path.write_text('')  # mark as missing
                break
        for path in paths:
            if path.stat().st_size == 0:
                missing.add(path)
    return missing


def layout(
        row: AtomsRow,
        key_descriptions: Dict[str, Tuple[str, str, str]],
        prefix: Path,
        pool: Optional[multiprocessing.Pool] = None
) -> List[Tuple[str, List[List[Dict[str, Any]]]]]:
    """
    Page layout for each ase.db row. This is only executed when the row's
    webpage is loaded into the row.html.
    """
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'large',
              'axes.titlesize': 'large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'savefig.dpi': 200}
    with plt.rc_context(params):
        return _layout(row, key_descriptions, prefix, pool)


def _layout(row, key_descriptions, prefix, pool):
    """
    Creates a list of web panel data in a tuple format, which indicates the
    panel title and web panel data.
    i.e. the content in the jinja2 row.html template
    """
    page = {}
    exclude = set()

    row = RowWrapper(row)

    newdata = parse_row_data(row.data)
    row.data = newdata
    result_objects = []

    for key, value in row.data.items():
        if isinstance(value, ASRResult):
            result_objects.append(value)
    panel_data_sources = {}
    # Locate all webpanels
    for result in result_objects:
        if 'ase_webpanel' not in result.get_formats():
            continue
        panels = result.format_as('ase_webpanel', row, key_descriptions)
        if not panels:
            continue

        for panel in panels:
            assert 'title' in panel, f'No title in {result} webpanel'
            if not isinstance(panel, WebPanel):
                panel = WebPanel(**panel)
            paneltitle = describe_entry(str(panel['title']), description='')

            if paneltitle in page:
                panel_data_sources[paneltitle].append(result)
                page[paneltitle].append(panel)
            else:
                panel_data_sources[paneltitle] = [result]
                page[paneltitle] = [panel]

    for paneltitle, data_sources in panel_data_sources.items():

        elements = []
        for panel in page[paneltitle]:
            tit = panel['title']
            if hasattr(tit, '__explanation__'):
                elements += [par(tit.__explanation__)]

        recipe_links = []
        for result in data_sources:
            asr_name = (result.metadata.asr_name
                        if 'asr_name' in result.metadata else '(Unknown data source)')
            link_name = HTMLStringFormat('').get_recipe_href(asr_name=asr_name)
            recipe_links.append(link_name)

        links = (bold("Relevant recipes")
                 + br
                 + 'This panel contains information calculated with '
                 'the following ASR Recipes:' + br + HTMLStringFormat.indent_lst(
                recipe_links))
        elements.append(par(links))
        description = HTMLStringFormat('').combine_elements(elements)
        describe_entry(paneltitle, description=description,
                       title='General panel information')

    merge_panels(page)
    page = [panel for _, panel in page.items()]
    # Sort sections if they have a sort key
    page = [x for x in sorted(page, key=lambda x: x.get('sort', 99))]

    subpanel = []
    for panel in page:
        subpanel.append(panel.subpanel)

    # add a miscellaneous section
    misc_title, misc_columns = miscellaneous_section(row, key_descriptions,
                                                     exclude)
    misc_panel = {'title': misc_title,
                  'columns': misc_columns}
    page.append(misc_panel)

    # add a links section
    link_title, link_columns = link_section(row, key_descriptions,
                                            exclude)
    link_panel = {'title': link_title,
                  'columns': link_columns}
    page.append(link_panel)

    # Get descriptions of figures that are created by all webpanels
    plot_descriptions = []
    for panel in page:
        plot_descriptions.extend(panel.get('plot_descriptions', []))

    # List of functions and the figures they create:
    missing_figures = generate_plots(row, prefix, plot_descriptions, pool)

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
        if Path(prefix + block['filename']) in missing_figures:
            return False
        return True

    # Remove missing figures from layout
    final_page = []
    for title, columns in asepage:
        columns = [[block for block in column if ok(block)]
                   for column in columns]
        if any(columns):
            final_page.append((title, columns))

    return final_page, subpanel


@command('asr.webpages.browser')
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
    # print(cmd)
    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit(1)


if __name__ == '__main__':
    main.cli()


def subpanels(final_page):
    names = [final_page[i][0] for i in range(len(final_page))]
