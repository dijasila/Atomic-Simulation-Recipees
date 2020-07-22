from typing import List
from asr.core import command, option


@command('asr.push',
         dependencies=['asr.structureinfo', 'asr.phonopy'])
@option('-q', '--momentum', nargs=3, type=float,
        help='Phonon momentum')
@option('-m', '--mode', type=int, help='Mode index')
@option('-a', '--amplitude', type=float,
        help='Maximum distance an atom will be displaced')
def main(momentum: List[float] = [0, 0, 0], mode: int = 0,
         amplitude: float = 0.1):
    """Push structure along some phonon mode and relax structure."""
    import numpy as np
    q_c = momentum

    # Get modes
    from ase.io import read
    from asr.core import read_json
    atoms = read('structure.json')
    data = read_json("results-asr.phonopy.json")

    u_klav = data["u_klav"]

    q_qc = data["q_qc"]

    diff_kc = np.array(list(q_qc)) - q_c 
    diff_kc -= np.round(diff_kc) 
    ind = np.argwhere(np.all(np.abs(diff_kc) < 1e-2, 1))[0,0]
    print(ind)

    #assert momentum in q_qc.tolist(), "No momentum in calculated q-points"
    # Repeat atoms
    from fractions import Fraction
    repeat_c = [Fraction(qc).limit_denominator(10).denominator for qc in q_c]
    newatoms = atoms * repeat_c
    print(repeat_c)

    # Repeat atoms 
    from fractions import Fraction 
    newatoms = atoms * repeat_c 
    # Here `Na` refers to a composite unit cell/atom dimension 
    pos_Nav = newatoms.get_positions() 
    # Total number of unit cells 
    N = np.prod(repeat_c) 
 
    # Corresponding lattice vectors R_m 
    R_cN = np.indices(repeat_c).reshape(3, -1) 
 
    # Bloch phase 
    phase_N = np.exp(2j * np.pi * np.dot(q_c, R_cN)) 
    phase_Na = phase_N.repeat(len(atoms)) 
    m_Na = newatoms.get_masses() 
 
    # Repeat and multiply by Bloch phase factor 
    mode_av = u_klav[ind, mode] 
    n_a = np.linalg.norm(mode_av, axis=1) 
    mode2_av = mode_av / np.max(n_a) 
    mode_Nav = (np.vstack(N * [mode2_av]) * phase_Na[:, np.newaxis] * amplitude / m_Na[:, np.newaxis])
    newatoms.set_positions(pos_Nav + mode_Nav.real)

    # Write unrelaxed.json file to folder
    folder = 'push-{}-q-{}-{}-{}-mode-{}'.format(amplitude, q_c[0], q_c[1], q_c[2], mode)
    from gpaw.mpi import world
    from pathlib import Path
    from ase.io import write
    if world.rank == 0 and not Path(folder).is_dir():
        Path(folder).mkdir()
    write(f'{folder}/unrelaxed.json', newatoms)


if __name__ == '__main__':
    main.cli()
