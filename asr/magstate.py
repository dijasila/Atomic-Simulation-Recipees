"""Module for determining magnetic state."""
from asr.core import command


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmom = calc.get_magnetic_moment()
    if abs(magmom) > 0.02:
        return 'fm'

    magmoms = calc.get_magnetic_moments()
    if abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
        return 'afm'

    # Material is essentially non-magnetic
    return 'nm'


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
    calc = GPAW('gs.gpw')
    magstate = get_magstate(calc)

    results = {'magstate': magstate.upper(),
               'is_magnetic': magstate != 'NM'}

    return results


if __name__ == '__main__':
    main.cli()
