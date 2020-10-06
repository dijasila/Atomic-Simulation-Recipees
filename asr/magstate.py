"""Module for determining magnetic state."""
from asr.core import command


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    maximum_mom = abs(magmoms).max()
    if magmoms is None or maximum_mom < 0.1:
        return 'nm'

    magmom = calc.get_magnetic_moment()

    if abs(magmom) < 0.01 and maximum_mom > 0.1:
        return 'afm'

    return 'fm'


def webpanel(row, key_descriptions):
    """Webpanel for magnetic state."""
    rows = [['Magnetic state', row.magstate]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': rows}]],
               'sort': 0}
    return [summary]


@command('asr.magstate',
         requires=['gs.gpw'],
         webpanel=webpanel,
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
