import asr
from ase.build import bulk
from ase.calculators.calculator import get_calculator_class
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter


@asr.instruction('asr.tutorial')
@asr.argument('crystal_structure', type=str)
@asr.argument('element', type=str)
def energy(element: str, crystal_structure: str) -> float:
    """Calculate the total energy per atom of atomic structure."""
    atoms = bulk(element, crystalstructure=crystal_structure, a=3.6)
    cls = get_calculator_class('emt')
    atoms.calc = cls()
    filt = ExpCellFilter(atoms, mask=[1, 1, 1, 1, 1, 1])
    opt = BFGS(filt, logfile=None)
    opt.run(fmax=0.001)
    energy = atoms.get_potential_energy()
    natoms = len(atoms)
    return round(energy / natoms, 6)


@asr.instruction('asr.tutorial')
@asr.argument('element', type=str)
def main(element: str) -> str:
    """Calculate lowest energy crystal structure for an element."""
    crystal_structures = [
        'sc', 'fcc', 'bcc', 'diamond',
    ]
    energies = []
    for crystal_structure in crystal_structures:
        en = energy(element, crystal_structure)
        energies.append(en)
    lowest_energy_crystal_structure = min(
        zip(crystal_structures, energies),
        key=lambda item: item[1]
    )[0]

    return lowest_energy_crystal_structure
