"""Push along phonon modes."""
# TODO: Should be moved to setup recipes.
from typing import List
from fractions import Fraction
import numpy as np

from ase import Atoms


#@command('asr.c2db.push')
#@atomsopt
#@calcopt
#@option('-q', '--momentum', nargs=3, type=float,
#        help='Phonon momentum')
#@option('-m', '--mode', type=int, help='Mode index')
#@option('-a', '--amplitude', type=float,
#        help='Maximum distance an atom will be displaced')
#@option('-n', help='Supercell size', type=int)
#@option('--mingo/--no-mingo', is_flag=True,
#        help='Perform Mingo correction of force constant matrix')
def main(
        atoms: Atoms,
        phresults,
        momentum: List[float] = [0, 0, 0],
        mode: int = 0,
        amplitude: float = 0.1,
) -> Atoms:
    """Push structure along some phonon mode and relax structure."""
    from asr.phonons import analyse
    q_c = momentum

    omega_kl = phresults.omega_kl
    u_klav = phresults.u_kl
    q_qc = phresults.q_qc
    omega_kl, u_klav, q_qc = analyse(modes=True, q_qc=[q_c])

    iq = np.argwhere(
        (np.sum(np.abs(q_qc - q_c), axis=0) < 1e-3).all(axis=1)
    )

    # Repeat atoms
    repeat_c = [Fraction(qc).denominator
                if qc > 1e-3 else 1 for qc in q_qc[iq]]
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
    mode_av = u_klav[iq, mode]
    n_a = np.linalg.norm(mode_av, axis=1)
    mode_av /= np.max(n_a)
    mode_Nav = (np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis] * amplitude)
    newatoms.set_positions(pos_Nav + mode_Nav.real)

    return newatoms
