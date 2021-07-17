from collections.abc import Mapping
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import traceback
import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from asr.core import decode_object, ASRResult
from asr.core.cache import Cache, MemoryBackend
from asr.core.datacontext import DataContext
from ase.db.row import AtomsRow
from ase.db.core import float_to_time_string, now
from .webpanel import WebPanel


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
                old_value = value
                value = '{:.{}f}'.format(value, digits)
                if hasattr(old_value, '__explanation__'):
                    value = describe_entry(value, old_value.__explanation__)
            elif not isinstance(value, str):
                value = str(value)

            longdesc, desc, unit = key_descriptions.get(key, ['', key, ''])
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
    return ('Miscellaneous', [[misc]])


def link_section(row, key_descriptions, exclude):
    """Make help function for adding a "links" section.

    Create table with all keys except those in exclude.
    """
    try:
        links = row.data['links']
    except KeyError:
        return ('Links', [[]])

    link_table = create_link_table(row, links, key_descriptions)
    return ('Links', [[link_table]])


def create_link_table(row, links, key_descriptions):
    """Create links table in the links panel."""
    link_table = table(row, 'Links', [])
    for link in links:
        linkname = f'<a href="{link[1]}">{link[0]}</a>'
        linktype = link[2]
        link_table['rows'].extend([[linkname, linktype]])

    return link_table


value_type_to_explained_type = {}


def describe_entry(value, description, title='Information'):
    """Describe website entry.

    This function sets an __explanation__ attribute on the given object
    which is used by the web application to generate additional explanations.
    """
    description = normalize_string(description)
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


def get_recipe_href(asr_name, name=None):
    """Get a hyperlink for the recipe documentation associated with a given result.

    Parameters
    ----------
    asr_name : str
        asr_name variable of recipe
    name : str/None
        name for link - falls back to asr_name if None

    Returns
    -------
    link_name : str
    """
    if name is None:
        name = asr_name
    # ATM href only works to recipe main
    asr_name = asr_name.split(':')[0]
    link_name = ('<a href="https://asr.readthedocs.io/en/latest/'
                 f'src/generated/recipe_{asr_name}.html">{name}</a>')

    return link_name


def make_html_tag_wrapper(tag):

    def wrap_tag(text):
        return f'<{tag}>{text}</{tag}>'

    return wrap_tag


def div(text, cls=''):
    return f'<div class="{cls}">{text}</div>'


def html_table(rows, header=None):
    text = '<table class="table">'
    if header is not None:
        headtext = ''
        for value in header:
            headtext += th(value)
        text += thead(headtext)
    for row in rows:
        rowtext = ''
        for value in row:
            rowtext += td(value)
        rowtext = tr(rowtext)
        text += rowtext
    text += '</table>'
    return text


li = make_html_tag_wrapper('li')
bold = make_html_tag_wrapper('b')
pre = make_html_tag_wrapper('pre')
code = make_html_tag_wrapper('code')
dt = make_html_tag_wrapper('dt')
dd = make_html_tag_wrapper('dd')
tr = make_html_tag_wrapper('tr')
td = make_html_tag_wrapper('td')
th = make_html_tag_wrapper('th')
thead = make_html_tag_wrapper('thead')
par = make_html_tag_wrapper('p')

br = '<br>'


def ul(items):

    text = ''
    for item in items:
        text += li(item)

    return '<ul>' + text + '</ul>'


def dl(items):

    text = ''
    for item1, item2 in items:
        text += dt(item1) + dd(item2)

    return '<dl class="dl-horizontal">' + text + '<dl>'


def href(text, link):
    return f'<a href="{link}">{text}</a>'


static_article_links = {'C2DB': href(
    """S. Haastrup et al. The Computational 2D Materials Database: high-throughput
modeling and discovery of atomically thin crystals, 2D Mater. 5 042002
(2018).""",
    'https://doi.org/10.1088/2053-1583/aacfc1'
)
}


def normalize_string(text):
    while text.endswith('\n'):
        text = text[:-1]
    while text.startswith('\n'):
        text = text[1:]
    while text.endswith(br):
        text = text[:-len(br)]
    while text.startswith(br):
        text = text[len(br):]
    return text


def make_panel_description(text, articles=None):

    if articles:
        articles = (
            bold('Relevant articles:')
            + ul([
                static_article_links.get(article, article) for article in articles]
            )
        )
        elements = [text, articles]
    else:
        elements = [text]

    return combine_elements(elements)


def combine_elements(items, spacer=(br + br)):
    items = [normalize_string(item) for item in items]

    return spacer.join(items)


def format_parameter_description(name, params, exclude_keys):
    link_name = get_recipe_href(name)
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


def table(row, title, keys, kd={}, digits=2):
    return create_table(row, [title, 'Value'], keys, kd, digits)


def make_bold(text: str) -> str:
    return f'<b>{text}</b>'


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
            rows[0][0] = make_bold(title)
        elif column_index > 0 and columnlabels is not None:
            rows[0][column_index] = make_bold(columnlabels[column_index - 1])

    for row_index in range(shape[0]):
        if row_index > 0:
            rows[row_index][0] = make_bold(rowlabels[row_index - 1])

    for i in range(1, shape[0]):
        for j in range(1, shape[1]):
            value = M[i - 1][j - 1]
            rows[i][j] = '{:.{}f}{}'.format(value, digits, unit)

    table = dict(type='table',
                 rows=rows)
    return table


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
        panel = WebPanel(**panel)
        page[title] = panel


def extract_recipe_from_filename(filename: str):
    """Parse filename and return recipe name."""
    pattern = re.compile('results-(.*)\.json')  # noqa
    m = pattern.match(filename)
    return m.group(1)


def is_results_file(filename):
    return filename.startswith('results-') and filename.endswith('.json')


class DataFilenameTranslator:

    def __init__(self, cache, data=None):
        if data is None:
            data = {}
        self.data = data
        self.cache = cache

    def __getitem__(self, item):
        if item in ['links']:
            return self.data[item]
        selector = self.filename_to_selector(item)
        records = self.cache.select(selector=selector)
        assert len(records) == 1
        record = records[0]
        return record.result

    def filename_to_selector(self, filename):
        sel = self.cache.make_selector()
        funcname = filename[8:-5]
        if '@' not in funcname:
            funcname += ':main'
        else:
            funcname = funcname.replace('@', ':')
        sel.run_specification.name = sel.EQ(funcname)
        return sel

    def get(self, item, default=None):
        if item in self:
            return self[item]
        return default

    def get_record(self, filename):
        selector = self.filename_to_selector(filename)
        records = self.cache.select(selector=selector)
        record = records[0]
        return record

    def __contains__(self, item):
        selector = self.filename_to_selector(item)
        return self.cache.has(selector=selector)


class RowWrapper(Mapping):

    def __init__(self, row):
        from asr.database.fromtree import serializer
        cache = Cache(backend=MemoryBackend())
        if 'records' in row.data:
            records = serializer.deserialize(row.data['records'])
        else:
            records = []
        self.records = records
        for record in records:
            cache.add(record)
        self._row = row
        self.cache = cache
        self.data = DataFilenameTranslator(cache, data=row.data)

    def __getstate__(self):
        return vars(self).copy()

    def __setstate__(self, dct):
        vars(self).update(dct)

    def __getattr__(self, key):
        """Wrap attribute lookup of AtomsRow."""
        return getattr(self._row, key)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return iter(self._row)

    def __len__(self):
        return len(self._row)

    @property
    def ctime(self):
        return self._row.ctime

    @property
    def key_value_pairs(self):
        return self._row.key_value_pairs


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


def generate_plots(context, prefix, plot_descriptions, pool):
    missing = set()
    for desc in plot_descriptions:
        function = desc['function']
        filenames = desc['filenames']
        paths = [Path(prefix + filename) for filename in filenames]
        for path in paths:
            if not path.is_file():
                # Create figure(s) only once:
                strpaths = [str(path) for path in paths]
                args = [function, context] + strpaths

                try:
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
        pool: Optional[multiprocessing.Pool] = None,
) -> List[Tuple[str, List[List[Dict[str, Any]]]]]:
    """Page layout."""
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'large',
              'axes.titlesize': 'large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'savefig.dpi': 200}
    with plt.rc_context(params):
        return _layout(row, key_descriptions, prefix, pool)


def _layout(row, key_descriptions, prefix, pool):
    page = {}
    exclude = set()

    row = RowWrapper(row)
#
#     newdata = parse_row_data(row.data)
#     row.data = newdata
#     result_objects = []
#
#     for key, value in row.data.items():
#         if isinstance(value, ASRResult):
#             result_objects.append(value)
    panel_data_sources = {}
    # Locate all webpanels

    for record in row.records:
        context = DataContext(row, record, row.cache)

        result = record.result
        if not isinstance(result, ASRResult):
            continue

        if 'webpanel2' not in result.formats:
            continue

        panels = result.format_as('webpanel2', context)

        if not panels:
            continue

        for panel in panels:
            assert 'title' in panel, f'No title in {result} webpanel'
            if not isinstance(panel, WebPanel):
                panel = WebPanel(**panel)
            paneltitle = describe_entry(str(panel['title']), description='')

            if paneltitle in page:
                panel_data_sources[paneltitle].append(record)
                page[paneltitle].append(panel)
            else:
                panel_data_sources[paneltitle] = [record]
                page[paneltitle] = [panel]

        # Get descriptions of figures that are created by all webpanels
        plot_descriptions = []
        for panel in panels:
            plot_descriptions.extend(panel.get('plot_descriptions', []))

        # List of functions and the figures they create:
        missing_figures = generate_plots(context, prefix, plot_descriptions,
                                         pool)

    for paneltitle, data_sources in panel_data_sources.items():

        elements = []
        for panel in page[paneltitle]:
            tit = panel['title']
            if hasattr(tit, '__explanation__'):
                elements += [par(tit.__explanation__)]

        recipe_links = []
        for record in data_sources:
            asr_name = record.run_specification.name

            link_name = get_recipe_href(asr_name)
            recipe_links.append(link_name)

        links = (bold("Relevant recipes")
                 + br
                 + 'This panel contains information calculated with '
                 'the following ASR Recipes:' + br + ul(recipe_links))
        elements.append(par(links))
        description = combine_elements(elements, spacer='')
        describe_entry(paneltitle, description=description,
                       title='General panel information')

    merge_panels(page)
    page = [panel for _, panel in page.items()]
    # Sort sections if they have a sort key
    page = [x for x in sorted(page, key=lambda x: x.get('sort', 99))]

    # add miscellaneous section
    misc_title, misc_columns = miscellaneous_section(row, key_descriptions,
                                                     exclude)
    misc_panel = {'title': misc_title,
                  'columns': misc_columns}
    page.append(misc_panel)

    # add links section
    link_title, link_columns = link_section(row, key_descriptions,
                                            exclude)
    link_panel = {'title': link_title,
                  'columns': link_columns}
    page.append(link_panel)

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

    # Remove missing figures from layout:
    final_page = []
    for title, columns in asepage:
        columns = [[block for block in column if ok(block)]
                   for column in columns]
        if any(columns):
            final_page.append((title, columns))
    return final_page


def get_attribute(obj, attrs):

    if not attrs:
        return obj

    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            try:
                obj = obj[attr]
            except (TypeError, KeyError):
                obj = None

    return obj


def parse_selectors(selector: str):
    op, attr = selector.split(',')
    attrs = attr.split('.')
    if op == '-':
        sign = -1
    else:
        sign = 1
    return sign, attrs


def get_panels_values(panels):
    for ip, panel in enumerate(panels):
        columns = panel['columns']
        for ic, column in enumerate(columns):
            for ie, element in enumerate(column):
                if element['type'] == 'table':
                    rows = element['rows']
                    for ir, row in enumerate(rows):
                        for iv, value in enumerate(row):
                            if iv > 0:
                                yield value, (ip, 'columns', ic, ie, 'rows', ir, iv)


def get_value(panels, indices):
    value = panels
    for ind in indices:
        value = value[ind]
    return value


def set_value(panels, indices, value):
    values = get_value(panels, indices[:-1])
    values[indices[-1]] = value


def cache_webpanel(recipename, *selectors):
    from asr.database.browser import html_table, par, br, bold

    def decorator(func):
        def wrapper(result, row, key_descriptions):
            cache = row.cache
            records = cache.select(name=recipename)

            sortattrs = []
            signs = []
            for selector in selectors:
                sign, attrs = parse_selectors(selector)

                signs.append(sign)
                sortattrs.append(
                    ['run_specification', 'parameters', *attrs]
                )

            def keysort(x):
                keys = []
                for sign, attrs in zip(signs, sortattrs):
                    value = get_attribute(x, attrs)
                    if value is not None:
                        keys.append(sign * value)
                    else:
                        keys.append(None)
                return keys

            records = sorted(
                records,
                key=keysort,
            )

            webpanels_r = []
            for record in records:
                webpanels_r.append(func(record.result, row, key_descriptions))

            representative_panels = webpanels_r[-1]
            for value, indices in get_panels_values(representative_panels):
                table = []
                header = ['.'.join(attrs[-2:]) for attrs in sortattrs] + ['value']
                for record, webpanels in zip(records, webpanels_r):
                    other_value = get_value(webpanels, indices)
                    row = (
                        [get_attribute(record, attrs) for attrs in sortattrs]
                        + [other_value]
                    )
                    table.append(row)

                # Most important entry on top
                table = table[::-1]
                html = par(bold('Convergence') + br
                           + html_table(table, header=header))
                value = describe_entry(value, description=html)
                set_value(webpanels, indices, value)

            return representative_panels
        return wrapper
    return decorator
