"""Module for generating atomic structures with displaced atoms."""

from ase import Atoms

import asr
from asr.core import command, option, ASRResult, atomsopt


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


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ('asr.setup.displacements:main')


@asr.migration(selector=sel)
def remove_copy_params_parameter(record):
    """Remove copy_params parameter."""
    del record.parameters.copy_params
    return record


@command(
    'asr.setup.displacements',
)
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
