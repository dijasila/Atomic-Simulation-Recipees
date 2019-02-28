def get_parser():
    import argparse
    desc = 'Push structure along some phonon mode and relax structure'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-q', '--momentum', default=[0, 0, 0],
                        nargs=3, type=float,
                        help='Phonon momentum')
    parser.add_argument('-m', '--mode', default=0, type=int,
                        help='Mode index')
    parser.add_argument('-a', '--amplitude', default=0.1, type=float,
                        help='Maximum distance an atom will be displaced')
    parser.add_argument('--fix-cell', action='store_true',
                        help='Do not relax cell')
    return parser
    

def main(args=None):
    from gpaw import GPAW
    from c3db.phonons import analyse2
    import numpy as np

    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    mode = args['mode']
    q_c = args['momentum']
    amplitude = args['amplitude']
    
    # Get modes
    calc = GPAW('gs.gpw', txt=None)
    atoms = calc.atoms
    omega_kl, u_klav, q_qc = analyse2(atoms, modes=True, q_qc=[q_c])

    # Repeat atoms
    from fractions import Fraction
    repeat_c = [Fraction(1 / qc).denominator if qc > 1e-3 else 1
                for qc in q_qc[0]]
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
    mode_Nav = (np.vstack(N * [mode_av]) *
                phase_Na[:, np.newaxis] * amplitude)
    newatoms.set_positions(pos_Nav + mode_Nav.real)

    from mcr.recipies.relax import relax
    tag = 'push-q=({},{},{})mode={}'.format(q_c[0], q_c[1], q_c[2],
                                            mode)
    smask = [1, 1, 1, 1, 1, 1]

    if args['fix_cell']:
        smask = [0, 0, 0, 0, 0, 0]
        tag += '-fix-cell'
    relax(newatoms, tag, smask=smask)

    
if __name__ == '__main__':
    main()
