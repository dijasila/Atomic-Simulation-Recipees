import pytest
from asr.webpages.appresources import HTMLStringFormat


@pytest.fixture
def text():
    return 'the text'


@pytest.fixture
def lst():
    return ['the cats run', 'dogs bark', 'rawr']


@pytest.fixture
def items():
    return [['the cats run', 'around the tree']]


def test_href(text):
    href_format = HTMLStringFormat.href(text=text, link='https')
    assert '<a href="https" target="_blank">the text</a>' == href_format


def test_div(text):
    div = HTMLStringFormat.div()(text=text, html_class='')
    div_str = '<div class="">the text</div>'
    assert div == div_str


def test_linebreak(text):
    br = HTMLStringFormat.linebr()(text=text)
    br_format = f'{text}<br>'
    assert br == br_format

    br = HTMLStringFormat.linebr()(text=text, end=False)
    br_format = f'<br>{text}'
    assert br == br_format


def test_bold(text):
    bold = HTMLStringFormat.bold()(text=text)
    b_format = f'<b>{text}</b>'
    assert bold == b_format


def test_par(text):
    par = HTMLStringFormat.par()(text=text)
    par_format = f'<p>{text}</p>'
    assert par == par_format


def test_lst(text):
    li = HTMLStringFormat.lst()(text)
    li_format = '<li>the text</li>'
    assert li == li_format


def test_indent_lst(lst):
    indent_list = HTMLStringFormat.indent_lst(items=lst)
    indent_list_format = '<ul class=""><li>the cats run</li>'\
                         '<li>dogs bark</li><li>rawr</li></ul>'
    assert indent_list == indent_list_format


def test_dt(text):
    dt = HTMLStringFormat.dt()(text=text)
    dt_format = f'<dt>{text}</dt>'
    assert dt == dt_format


def test_dd(text):
    dd = HTMLStringFormat.dd()(text=text)
    dd_format = f'<dd>{text}</dd>'
    assert dd == dd_format


def test_descriptivelist(items):
    dl = HTMLStringFormat.dlst(items=items)
    dl_str = '<dl class="dl-horizontal"><dt>the cats run</dt>'\
             '<dd>around the tree</dd></dl>'
    assert dl == dl_str


def test_indented_descriptivelist(lst):
    indent_lst = HTMLStringFormat.indent_lst(items=lst)
    assert (indent_lst == '<ul class=""><li>the cats run</li>'
                          '<li>dogs bark</li><li>rawr</li></ul>')
