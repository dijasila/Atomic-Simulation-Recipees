from asr.utils import command, option


@command('asr.push')
@option(
    '-q',
    '--momentum',
    default=[0, 0, 0],
    nargs=3,
    type=float,
    help='Phonon momentum')
@option('-m', '--mode', default=0, type=int, help='Mode index')
@option(
    '-a',
    '--amplitude',
    default=0.1,
    type=float,
    help='Maximum distance an atom will be displaced')
@option('--fix-cell', is_flag=True, help='Do not relax cell')
@option('--show-mode', is_flag=True, help='Save mode to tmp.traj for viewing')
@option('-n', default=2, help='Supercell size')
def main(momentum, mode, amplitude, fix_cell, show_mode, n):
    """Push structure along some phonon mode and relax structure"""
    from asr.phonons import analyse
    import numpy as np
    q_c = momentum

    # Get modes
    from ase.io import read
    atoms = read('start.json')
    omega_kl, u_klav, q_qc = analyse(atoms, modes=True, q_qc=[q_c], N=n)

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

    from asr.relax import relax
    tag = 'push-q-{}-{}-{}-mode-{}'.format(q_c[0], q_c[1], q_c[2], mode)
    smask = None
    if fix_cell:
        smask = [0, 0, 0, 0, 0, 0]
        tag += '-fix-cell'

    if show_mode:
        from ase.io.trajectory import Trajectory
        traj = Trajectory('tmp.traj', mode='w')
        showatoms = newatoms.copy()
        n = 20
        phase_x = np.linspace(-1, 1, n) * np.pi
        for i, phase in enumerate(phase_x):
            amp = np.sin(phase) * amplitude
            showatoms.set_positions(pos_Nav + mode_Nav.real * amp)
            traj.write(showatoms)

        return
    relaxed = relax(newatoms, tag, smask=smask)

    # Write start.start file to folder
    name = tag + '/start.json'
    from gpaw.mpi import world
    from pathlib import Path
    from ase.io import write
    if world.rank == 0 and not Path(tag).is_dir():
        Path(tag).mkdir()
    write(name, relaxed)


dependencies = ['asr.phonons']
group = 'Structure'

if __name__ == '__main__':
    main()
