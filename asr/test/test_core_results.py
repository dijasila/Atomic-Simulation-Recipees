from typing import Dict
from asr.core import ASRResults, set_docstring


@set_docstring
class MyResults(ASRResults):
    """Generic results."""

    a: int

    key_descriptions: Dict[str, str] = {'a': 'A description of "a".'}


def test_results_object(capsys):
    results = MyResults(a=1)
    assert results.a == 1
    assert 'a' in results
    assert results.__doc__ == '\n'.join(['Generic results.',
                                         '',
                                         'Parameters',
                                         '----------',
                                         'a: <class \'int\'>',
                                         '    A description of "a".'])

    print(results)
    captured = capsys.readouterr()
    assert captured.out == 'a=1\n'

    assert isinstance(results.format_as('ase_webpanel'), list)

    html = results.format_as('html')
    html2 = format(results, 'html')
    assert html == html2
    assert f'{results:html}' == html

    format(results, 'json')
