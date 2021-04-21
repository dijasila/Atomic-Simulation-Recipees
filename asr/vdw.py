"""Module containing functionality for vdW properties."""
from asr.core import command, option, DictStr, AtomsFile
from ase import Atoms


@command('asr.vdw')
@option('--atoms', type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator specification.',
        type=DictStr())
def main(atoms: Atoms, calculator: dict = {'name': 'dftd3'}):
    """Calculate van der Waals energy contribution.

    Parameters
    ----------
    atoms: Atoms
        Atomic structure.
    calculator: dict
        Calculator specification.

    Returns
    -------
    dict
        Dict containing key "vdw_energy" with the van der waals energy.
    """
    from ase.calculators.calculator import get_calculator_class

    accepted_calculators = {'dftd3'}
    name = calculator.pop('name')
    assert name in accepted_calculators
    Calculator = get_calculator_class(name)
    calc = Calculator(**calculator)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    results = {'vdw_energy': energy}
    return results
