"""Module for determining magnetic state."""
from asr.core import command, ASRResult


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    if magmoms is None or abs(magmoms).max() < 0.1:
        return 'nm'

    maxmom = magmoms.max()
    minmom = magmoms.min()
    if abs(magmoms).max() >= 0.1 and \
       abs(maxmom - minmom) < abs(maxmom):
        return 'fm'

    return 'afm'


def webpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    rows = [['Magnetic state', row.magstate]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': rows}]],
               'sort': 0}
    return [summary]


class Result(ASRResult):

    formats = {"ase_webpanel": webpanel}


@command('asr.magstate',
         requires=['gs.gpw'],
         returns=Result,
         dependencies=['asr.gs@calculate'])
def main():
    """Determine magnetic state."""
    from gpaw import GPAW
    calc = GPAW('gs.gpw', txt=None)
    magstate = get_magstate(calc)
    magmoms = calc.get_property('magmoms', allow_calculation=False)
    nspins = calc.get_number_of_spins()
    results = {'magstate': magstate.upper(),
               'is_magnetic': magstate != 'nm',
               'magmoms': magmoms,
               'nspins': nspins}

    return results


if __name__ == '__main__':
    main.cli()
