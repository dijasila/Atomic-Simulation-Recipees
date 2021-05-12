import asr
from ase.calculators.calculator import get_calculator_class


@asr.instruction('asr.tutorial')
@asr.atomsopt
def energy(atoms):
    """Calculate the total energy per atom of atomic structure."""
    cls = get_calculator_class('emt')
    atoms.calc = cls()
    energy = atoms.get_potential_energy()
    natoms = len(atoms)
    return energy / natoms
