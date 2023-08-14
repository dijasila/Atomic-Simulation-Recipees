from typing import List
from asr.core import command, ASRResult, prepare_result


def webpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    from asr.database.browser import WebPanel

    if row.get('magstate', 'NM') == 'NM':
        return []
    else:
        magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', f'{orbmag:.3f}']
                        for a, (symbol, magmom, orbmag)
                        in enumerate(zip(row.get('symbols'),
                                         result.magmom_a,
                                         result.orbmag_a))]

        unit = '(μ<sub>B</sub>)'
        magmoms_table = {'type': 'table',
                         'header': ['Atom index', 'Atom type',
                                    'Local spin magnetic moment ' + unit,
                                    'Local orbital magnetic moment ' + unit],
                         'rows': magmoms_rows}

        from asr.utils.hacks import gs_xcname_from_row
        xcname = gs_xcname_from_row(row)
        panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                         columns=[[], [magmoms_table]],
                         sort=11)

        return [panel]


@prepare_result
class Result(ASRResult):

    magmom_a: List[float]
    orbmag_a: List[float]

    key_descriptions = {
        "magmom_a": "Local spin magnetic moments [μ_B]",
        "orbmag_a": "Local orbital magnetic moments [μ_B]"
    }

    formats = {"ase_webpanel": webpanel}


@command('asr.collect_magmoms',
         returns=Result,
         dependencies=['asr.magstate', 'asr.orbmag'])
def main() -> Result:
    """Calculate local orbital magnetic moments."""

    from asr.core import read_json
    magstate = read_json('results-asr.magstate.json')['magstate']

    results = {}
    if magstate == 'NM':
        results['magmom_a'] = 0
        results['orbmag_a'] = 0
        return Result(data=results)

    orbmag_a = read_json('results-asr.orbmag.json')['orbmag_a']
    magmom_a = read_json('results-asr.magstate.json')['magmoms']
    results = {'magmom_a': magmom_a,
               'orbmag_a': orbmag_a}

    return Result(data=results)


if __name__ == '__main__':
    main.cli()
