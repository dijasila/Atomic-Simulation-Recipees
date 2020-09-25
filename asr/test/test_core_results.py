from typing import Dict
from asr.core import ASRResult, set_docstring, WebPanelEncoder, command


class MyWebPanel(WebPanelEncoder):
    """WebPanel for testing."""

    pass


webpanel = WebPanelEncoder()


@set_docstring
class MyResultVer0(ASRResult):
    """Generic results."""

    a: int
    version: int = 0
    key_descriptions: Dict[str, str] = {'a': 'A description of "a".'}


@set_docstring
class MyResult(ASRResult):
    """Generic results."""

    a: int
    prev_version = MyResultVer0
    version: int = 1
    key_descriptions: Dict[str, str] = {'a': 'A description of "a".'}
    formats = {'ase_webpanel': webpanel}


@command('test_core_results',
         returns=MyResult)
def recipe():
    return MyResult()


def test_results_object(capsys):
    results = MyResult(a=1)
    results.metadata = {'time': 'right now'}
    assert results.a == 1
    assert 'a' in results
    assert results.__doc__ == '\n'.join(['Generic results.',
                                         '',
                                         'Data attributes',
                                         '---------------',
                                         'a: <class \'int\'>',
                                         '    A description of "a".'])

    formats = results.get_formats()
    assert formats['ase_webpanel'] == webpanel
    assert set(formats) == set('json', 'html', 'dict', 'ase_webpanel')
    print(results)
    captured = capsys.readouterr()
    assert captured.out == 'a=1\n'

    assert isinstance(results.format_as('ase_webpanel'), list)

    html = results.format_as('html')
    html2 = format(results, 'html')
    assert html == html2
    assert f'{results:html}' == html

    json = format(results, 'json')
    newresults = MyResult.from_format(json, format='json')
    assert newresults == results

    otherresults = MyResult(a=2)
    assert not otherresults == results


def test_reading_result():
    result = MyResult()
    parse_json(result)
    jsonresult = result.format_as('json')

    Resultsclass = recipe.returns

    Results.from_format(jsonresult, 'json')
