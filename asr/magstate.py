"""Module for determining magnetic state."""
from asr.core import (command, ASRResult, prepare_result, option,
                      AtomsFile, DictStr)
from asr.calculators import set_calculator_hook
from ase import Atoms
import typing


def get_magstate(calc):
    """Determine the magstate of calc."""
    magmoms = calc.get_property('magmoms', allow_calculation=False)

    if magmoms is None:
        return 'nm'

    maximum_mom = abs(magmoms).max()
    if maximum_mom < 0.1:
        return 'nm'

    magmom = calc.get_magnetic_moment()

    if abs(magmom) < 0.01 and maximum_mom > 0.1:
        return 'afm'

    return 'fm'


def webpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    rows = [['Magnetic state', row.magstate]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': rows}]],
               'sort': 0}
    return [summary]


@prepare_result
class Result(ASRResult):

    magstate: str
    is_magnetic: bool
    magmoms: typing.List[float]
    magmom: float
    nspins: int

    key_descriptions = {'magstate': 'Magnetic state.',
                        'is_magnetic': 'Is the material magnetic?',
                        'magmoms': 'Atomic magnetic moments.',
                        'magmom': 'Total magnetic moment.',
                        'nspins': 'Number of spins in system.'}
    formats = {"ase_webpanel": webpanel}


@command('asr.magstate',
         returns=Result,
         argument_hooks=[set_calculator_hook])
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def main(atoms: Atoms,
         calculator: dict = {
             'name': 'gpaw',
             'mode': {'name': 'pw', 'ecut': 800},
             'xc': 'PBE',
             'basis': 'dzp',
             'kpts': {'density': 12.0, 'gamma': True},
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'convergence': {'bands': 'CBM+3.0'},
             'nbands': '200%',
             'txt': 'gs.txt',
             'charge': 0
         }) -> Result:
    """Determine magnetic state."""
    from asr.gs import calculate as calculategs

    calculaterecord = calculategs(atoms=atoms, calculator=calculator)
    calc = calculaterecord.result.calculation.load()
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
