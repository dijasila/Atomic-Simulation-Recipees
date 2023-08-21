"""Module for determining magnetic state."""

from typing import List
from asr.core import command, ASRResult, prepare_result

atomic_mom_threshold = 0.1
total_mom_threshold = 0.1


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    if magmoms is None:
        return 'nm'

    maximum_mom = abs(magmoms).max()
    if maximum_mom < atomic_mom_threshold:
        return 'nm'

    magmom = calc.get_magnetic_moment()

    if abs(magmom) < total_mom_threshold and maximum_mom > atomic_mom_threshold:
        return 'afm'

    return 'fm'


def webpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    from asr.database.browser import describe_entry, dl, code, WebPanel

    is_magnetic = describe_entry(
        'Magnetic',
        'Is material magnetic?'
        + dl(
            [
                [
                    'Magnetic',
                    code('if max(abs(atomic_magnetic_moments)) > '
                         f'{atomic_mom_threshold}')
                ],
                [
                    'Not magnetic',
                    code('otherwise'),
                ],
            ]
        )
    )

    yesno = ['No', 'Yes'][row.is_magnetic]

    rows = [[is_magnetic, yesno]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Basic properties', ''],
                             'rows': rows}]],
               'sort': 0}

    """
    It makes sense to write the local orbital magnetic moments in the same
    table as the previous local spin magnetic moments; however, orbmag.py was
    added much later than magstate.py, so in order to accomplish this without
    inconvenient changes that may affect other people's projects, we need to
    load the orbmag.py results in this 'hacky' way
    """
    results_orbmag = row.data.get('results-asr.orbmag.json')
    if result.magstate == 'NM':
        return [summary]
    else:
        magmoms_header = ['Atom index', 'Atom type',
                          'Local spin magnetic moment (μ<sub>B</sub>)',
                          'Local orbital magnetic moment (μ<sub>B</sub>)']
        if results_orbmag is None:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', '--']
                            for a, (symbol, magmom)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms))]
        else:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', f'{orbmag:.3f}']
                            for a, (symbol, magmom, orbmag)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms,
                                             results_orbmag['orbmag_a']))]

        magmoms_table = {'type': 'table',
                         'header': magmoms_header,
                         'rows': magmoms_rows}

        from asr.utils.hacks import gs_xcname_from_row
        xcname = gs_xcname_from_row(row)
        panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                         columns=[[], [magmoms_table]], sort=11)

        return [summary, panel]


@prepare_result
class Result(ASRResult):

    magstate: str
    is_magnetic: bool
    magmoms: List[float]
    magmom: float
    nspins: int

    key_descriptions = {'magstate': 'Magnetic state.',
                        'is_magnetic': 'Is the material magnetic?',
                        'magmoms': 'Atomic magnetic moments.',
                        'magmom': 'Total magnetic moment.',
                        'nspins': 'Number of spins in system.'}

    formats = {"ase_webpanel": webpanel}


@command('asr.magstate',
         requires=['gs.gpw'],
         returns=Result,
         dependencies=['asr.gs@calculate'])
def main() -> Result:
    """Determine magnetic state."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)
    magstate = get_magstate(calc)
    magmoms = calc.get_property('magmoms', allow_calculation=False)
    magmom = calc.get_property('magmom', allow_calculation=False)
    nspins = calc.get_number_of_spins()
    results = {'magstate': magstate.upper(),
               'is_magnetic': magstate != 'nm',
               'magmoms': magmoms,
               'magmom': magmom,
               'nspins': nspins}

    return Result(data=results)


if __name__ == '__main__':
    main.cli()
