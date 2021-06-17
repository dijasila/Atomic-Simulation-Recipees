"""Generate magnetic atomic structures."""
from ase import Atoms

from asr.core import command, option, atomsopt


@command('asr.setup.magnetize')
@atomsopt
@option('--state', type=str,
        help='Comma separated string of magnetic states to create.')
def main(
        atoms: Atoms,
        state: str = 'nm',
) -> Atoms:
    """Set up atomic magnetic moments."""
    from asr.utils import magnetic_atoms
    import numpy as np
    msg = f'{state} is not a known state!'
    atoms = atoms.copy()

    # Non-magnetic:
    if state == 'nm':
        atoms.set_initial_magnetic_moments(None)
    elif state == 'fm':
        atoms.set_initial_magnetic_moments([1] * len(atoms))
    elif state == 'afm':
        magnetic = magnetic_atoms(atoms)
        nmag = sum(magnetic)
        if nmag == 2:
            magmoms = np.zeros(len(atoms))
            a1, a2 = np.where(magnetic)[0]
            magmoms[a1] = 1.0
            magmoms[a2] = -1.0
            atoms.set_initial_magnetic_moments(magmoms)
        else:
            raise ValueError(
                'Cannot produce afm state. '
                f'The number of magnetic atoms is {nmag}. '
                'At the moment, I only know how to do AFM '
                'state for 2 magnetic atoms.'
            )
    else:
        raise ValueError(msg)
    return atoms


if __name__ == '__main__':
    main.cli()
