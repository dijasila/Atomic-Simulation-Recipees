"""Module for generating atomic structures with displaced atoms."""

from ase import Atoms

from pathlib import Path
from asr.core import command, option, ASRResult, atomsopt


def get_displacement_folder(atomic_index,
                            cartesian_index,
                            displacement_sign,
                            displacement):
    """Generate folder name from (ia, iv, sign, displacement)."""
    cartesian_symbol = 'xyz'[cartesian_index]
    displacement_symbol = ' +-'[displacement_sign]
    foldername = (f'{displacement}-{atomic_index}'
                  f'-{displacement_symbol}{cartesian_symbol}')
    folder = Path('displacements') / foldername
    return folder


def create_displacements_folder(folder):
    folder.mkdir(parents=True, exist_ok=False)


def get_all_displacements(atoms):
    """Generate ia, iv, sign for all displacements."""
    for ia in range(len(atoms)):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)


def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms


@command('asr.setup.displacements')
@atomsopt
@option('--displacement', help='How much to displace atoms.', type=float)
def main(
        atoms: Atoms,
        displacement: float
) -> ASRResult:
    """Generate atomic displacements.

    Generate atomic structures with displaced atoms.
    """
    displaced_atoms = []
    for ia, iv, sign in get_all_displacements(atoms):
        new_structure = displace_atom(atoms, ia, iv, sign, displacement)
        displaced_atoms.append((ia, iv, sign, new_structure))

    return displaced_atoms
