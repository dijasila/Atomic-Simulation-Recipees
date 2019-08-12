from asr.utils import command, option


def sfrac(f):
    from gpaw.symmetry import frac
    if f == 0:
        return '0'
    return '%d/%d' % frac(f, n=2 * 3 * 4 * 5 * 6)


def print_symmetries(symmetry):
    n = len(symmetry.op_scc)
    nft = symmetry.ft_sc.any(1).sum()
    lines = ['Symmetries present (total): {0}'.format(n)]
    if not symmetry.symmorphic:
        lines.append(
            'Symmetries with fractional translations: {0}'.format(nft))

    # X-Y grid of symmetry matrices:

    lines.append('')
    nx = 6 if symmetry.symmorphic else 3
    ns = len(symmetry.op_scc)
    y = 0
    for y in range((ns + nx - 1) // nx):
        for c in range(3):
            line = ''
            for x in range(nx):
                s = x + y * nx
                if s == ns:
                    break
                op_c = symmetry.op_scc[s, c]
                ft = symmetry.ft_sc[s, c]
                line += '  (%2d %2d %2d)' % tuple(op_c)
                if not symmetry.symmorphic:
                    try:
                        line += ' + (%9s)' % sfrac(ft)
                    except ValueError:
                        line += f' + ({ft:.3e})'
            lines.append(line)
        lines.append('')
    return '\n'.join(lines)


def atomstospgcell(atoms, magmoms=None):
    from ase.calculators.calculator import PropertyNotImplementedError
    lattice = atoms.get_cell().array
    positions = atoms.get_scaled_positions(wrap=False)
    numbers = atoms.get_atomic_numbers()
    if magmoms is None:
        try:
            magmoms = atoms.get_magnetic_moments()
        except (RuntimeError, PropertyNotImplementedError):
            magmoms = None
    if magmoms is not None:
        return (lattice, positions, numbers, magmoms)
    return (lattice, positions, numbers)


def symmetrize_atoms(atoms, tolerance=None,
                     angle_tolerance=None,
                     return_dataset=False):
    import spglib
    import numpy as np
    from ase import Atoms
    spgcell = atomstospgcell(atoms)
    dataset = spglib.get_symmetry_dataset(spgcell, symprec=tolerance,
                                          angle_tolerance=angle_tolerance)
    cell_cv = spgcell[0]
    spos_ac = spgcell[1]
    numbers = spgcell[2]
    if len(spgcell) > 3:
        magmoms_a = spgcell[3]
    else:
        magmoms_a = None

    uspos_sac = []
    M_scc = []
    origin_sc = []
    point_c = np.zeros(3, float)
    U_scc = dataset['rotations']
    t_sc = dataset['translations']
    # t_sc -= np.rint(t_sc)
    for U_cc, t_c in zip(U_scc, t_sc):
        origin_sc.append(np.dot(U_cc, point_c) + t_c)
        symspos_ac = np.dot(spos_ac, U_cc.T) + t_c

        symcell_cv = np.dot(U_cc.T, cell_cv)

        # Cell metric
        M_cc = np.dot(symcell_cv, symcell_cv.T)
        M_scc.append(M_cc)
        inds = []
        for i, s_c in enumerate(spos_ac):
            d_ac = s_c - symspos_ac
            dm_ac = np.abs(d_ac - np.round(d_ac))
            ind = np.argwhere(np.all(dm_ac < tolerance, axis=1))[0][0]
            symspos_ac[ind] += np.round(d_ac[ind])
            inds.append(ind)
            assert atoms.numbers[i] == atoms.numbers[ind]

        assert len(set(inds)) == len(atoms)
        uspos_ac = symspos_ac[inds]
        uspos_sac.append(uspos_ac)

        assert np.all(np.abs(spos_ac - uspos_ac) < tolerance)

    origin_sc = np.array(origin_sc)
    origin_sc -= np.rint(t_sc)
    origin_c = np.mean(origin_sc, axis=0)
    # Shift origin
    print('Origin shifted by', origin_c)
    spos_ac = np.mean(uspos_sac, axis=0) - origin_c
    M_cc = np.mean(M_scc, axis=0)

    from ase.geometry.cell import cellpar_to_cell
    dotprods = M_cc[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    l_c = np.sqrt(dotprods[:3])
    angles_c = np.arccos(dotprods[3:] / l_c[[1, 0, 0]] / l_c[[2, 2, 1]])
    angles_c *= 180 / np.pi

    cp = np.concatenate([l_c, angles_c])

    ab_normal = np.cross(atoms.cell[0], atoms.cell[1])
    cell = cellpar_to_cell(cp, ab_normal=ab_normal,
                           a_direction=atoms.cell[0])

    idealized = Atoms(numbers=numbers,
                      scaled_positions=spos_ac, cell=cell,
                      pbc=True, magmoms=magmoms_a)

    newdataset = spglib.get_symmetry_dataset(atomstospgcell(idealized),
                                             symprec=tolerance,
                                             angle_tolerance=angle_tolerance)
    if return_dataset:
        return idealized, origin_c, dataset, newdataset

    return idealized, origin_c
    

@command('asr.setup.symmetrize',
         save_results_file=False)
@option('--tolerance', type=float,
        help='Tolerance when evaluating symmetries')
@option('--angle-tolerance', type=float,
        help='Tolerance one angles when evaluating symmetries')
def main(tolerance=1e-3, angle_tolerance=0.1):
    """Symmetrize atomic structure.

    This function changes the atomic positions and the unit cell
    of an approximately symmetric structure into an exactly
    symmetric structure.

    In practice, the spacegroup of the structure located in 'original.json'
    is evaluated using a not-very-strict tolerance, which can be adjusted using
    the --tolerance and --angle-tolerance switches. Then the symmetries of the
    spacegroup are used to generate equivalent atomic structures and by taking
    an average of these atomic positions we generate an exactly symmetric
    atomic structure.

    \b
    Examples:
    ---------
    Symmetrize an atomic structure using the default tolerances
        asr run setup.symmetrize
    """
    import numpy as np
    from ase.io import read, write
    atoms = read('original.json')

    assert atoms.pbc.all(), \
        ('Symmetrization has only been tested for 3D systems! '
         'To apply it to other systems you will have to test and update '
         'the code.')
    idealized = atoms.copy()
    spgs = []
    # There is a chance that the space group changes when symmetrizing
    # structure.
    maxiter = 2
    for i in range(maxiter):
        atol = angle_tolerance
        idealized, origin_c, dataset1, dataset2 = \
            symmetrize_atoms(idealized,
                             tolerance=tolerance,
                             angle_tolerance=atol,
                             return_dataset=True)
        spg1 = '{} ({})'.format(dataset1['international'],
                                dataset1['number'])
        spg2 = '{} ({})'.format(dataset2['international'],
                                dataset2['number'])
        if i == 0:
            spgs.extend([spg1, spg2])
        else:
            spgs.append(spg2)

        if spg1 == spg2:
            break
        print(f'Spacegroup changed {spg1} -> {spg2}. Trying again.')
    else:
        msg = 'Reached maximum iteration! Went through ' + ' -> '.join(spgs)
        raise RuntimeError(msg)
    print(f'Idealizing structure into spacegroup {spg2} using SPGLIB.')
    write('unrelaxed.json', idealized)

    # Check that the cell was only slightly perturbed
    nsym = len(dataset2['rotations'])
    cp = atoms.cell.cellpar()
    idcp = idealized.cell.cellpar()
    deltacp = idcp - cp
    abc, abg = deltacp[:3], deltacp[3:]

    print('Cell Change: (Δa, Δb, Δc, Δα, Δβ, Δγ) = '
          f'({abc[0]:.1e} Å, {abc[1]:.1e} Å, {abc[2]:.1e} Å, '
          f'{abg[0]:.2e}°, {abg[1]:.2e}°, {abg[2]:.2e}°)')

    assert (np.abs(abc) < 10 * tolerance).all(), \
        'a, b and/or c changed too much! See output above.'
    assert (np.abs(abg[3:]) < 10 * angle_tolerance).all(), \
        'α, β and/or γ changed too much! See output above.'

    cell = idealized.get_cell()
    spos_ac = atoms.get_scaled_positions(wrap=False)
    idspos_ac = idealized.get_scaled_positions(wrap=False) + origin_c
    dpos_av = np.dot(idspos_ac - spos_ac, cell)
    dpos_a = np.sqrt(np.sum(dpos_av**2, 1))
    with np.printoptions(precision=2, suppress=False):
        print(f'Change of positions:')
        msg = '    '
        for symbol, dpos in zip(atoms.symbols, dpos_a):
            msg += f' {symbol}: {dpos:.1e} Å,'
            if len(msg) > 70:
                print(msg[:-1])
                msg = '    '
        print(msg[:-1])

    assert (dpos_a < 10 * tolerance).all(), \
        'Some atoms moved too much! See output above.'

    # Check that GPAW indeed does find the same symmetries as
    # spglib
    from gpaw.symmetry import Symmetry
    id_a = idealized.get_atomic_numbers()
    symmetry = Symmetry(id_a, idealized.cell, idealized.pbc,
                        symmorphic=False,
                        time_reversal=False)
    symmetry.analyze(idealized.get_scaled_positions())

    nsymgpaw = len(symmetry.op_scc)

    if not nsym == nsymgpaw:
        msg = ('GPAW not finding as many symmetries as SPGLIB.\n'
               f'nsymgpaw={nsymgpaw} nsymspglib={nsym}\n')
        
        msg += 'GPAW symmetries:\n'
        msg += symmetry.__str__()

        msg += '\nSPGLIB symmetries:\n'
        # Monkey patch object
        spgsym = Symmetry(id_a, idealized.cell,
                          symmorphic=False,
                          time_reversal=False)

        spgsym.op_scc = np.array([op_cc for op_cc in dataset2['rotations']])
        spgsym.ft_sc = np.array([t_c for t_c in dataset2['translations']])
        msg += print_symmetries(spgsym)
        raise AssertionError(msg)


group = 'setup'


if __name__ == '__main__':
    main.cli()
