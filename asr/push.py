"""Push along phonon modes."""
# TODO: Should be moved to setup recipes.
from typing import List
from asr.core import command, option, ASRResult


@command('asr.push',
         dependencies=['asr.structureinfo', 'asr.phonons'])
@option('-q', '--momentum', nargs=3, type=float,
        help='Phonon momentum')
@option('-m', '--mode', type=int, help='Mode index')
@option('-a', '--amplitude', type=float,
        help='Maximum distance an atom will be displaced')
def main(momentum: List[float] = [0, 0, 0], mode: int = 0,
         amplitude: float = 0.1) -> ASRResult:
    """Push structure along some phonon mode and relax structure."""
    from asr.phonons import analyse
    import numpy as np
    q_c = momentum

    # Get modes
    from ase.io import read
    atoms = read('structure.json')
    omega_kl, u_klav, q_qc = analyse(modes=True, q_qc=[q_c])

    # Repeat atoms
    from fractions import Fraction
    repeat_c = [Fraction(qc).denominator if qc > 1e-3 else 1 for qc in q_qc[0]]
    newatoms = atoms * repeat_c

    # Here ``Na`` refers to a composite unit cell/atom dimension
    pos_Nav = newatoms.get_positions()

    # Total number of unit cells
    N = np.prod(repeat_c)

    # Corresponding lattice vectors R_m
    R_cN = np.indices(repeat_c).reshape(3, -1)

    # Bloch phase
    phase_N = np.exp(2j * np.pi * np.dot(q_c, R_cN))
    phase_Na = phase_N.repeat(len(atoms))

    # Repeat and multiply by Bloch phase factor
    mode_av = u_klav[0, mode]
    n_a = np.linalg.norm(mode_av, axis=1)
    mode_av /= np.max(n_a)
    mode_Nav = (np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis] * amplitude)
    newatoms.set_positions(pos_Nav + mode_Nav.real)

    # Write unrelaxed.json file to folder
    folder = 'push-q-{}-{}-{}-mode-{}'.format(q_c[0], q_c[1], q_c[2], mode)
    from gpaw.mpi import world
    from pathlib import Path
    from ase.io import write
    if world.rank == 0 and not Path(folder).is_dir():
        Path(folder).mkdir()
    write(f'{folder}/unrelaxed.json', newatoms)


if __name__ == '__main__':
    main.cli()
