"""Generate symmetrized atomic structure."""
from ase import Atoms
from asr.core import command, option, atomsopt


def symmetrize_atoms(atoms, tolerance=None,
                     angle_tolerance=None,
                     return_dataset=False):
    import numpy as np
    from ase import Atoms
    from asr.utils.symmetry import atoms2symmetry
    symmetry = atoms2symmetry(atoms, tolerance=tolerance,
                              angle_tolerance=angle_tolerance)
    dataset = symmetry.dataset
    cell_cv = atoms.get_cell()
    spos_ac = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

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
                      pbc=True)
    newsymmetry = atoms2symmetry(idealized, tolerance=tolerance,
                                 angle_tolerance=angle_tolerance)
    newdataset = newsymmetry.dataset
    if return_dataset:
        return idealized, origin_c, dataset, newdataset

    return idealized, origin_c


# XXX Originally read atoms from original.json.
@command('asr.setup.symmetrize')
@atomsopt(default='original.json')
@option('--tolerance', type=float,
        help='Tolerance when evaluating symmetries')
@option('--angle-tolerance', type=float,
        help='Tolerance one angles when evaluating symmetries')
def main(
        atoms: Atoms,
        tolerance: float = 1e-3,
        angle_tolerance: float = 0.1,
) -> Atoms:
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

    Examples
    --------
    Symmetrize an atomic structure using the default tolerances
    $ ase build -x diamond C original.json
    $ asr run setup.symmetrize

    """
    import numpy as np

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
    print(f'Idealizing structure into spacegroup {spg2}.')
    idealized.set_initial_magnetic_moments(
        atoms.get_initial_magnetic_moments())

    # Check that the cell was only slightly perturbed
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
        print('Change of positions:')
        msg = '    '
        for symbol, dpos in zip(atoms.symbols, dpos_a):
            msg += f' {symbol}: {dpos:.1e} Å,'
            if len(msg) > 70:
                print(msg[:-1])
                msg = '    '
        print(msg[:-1])

    assert (dpos_a < 10 * tolerance).all(), \
        'Some atoms moved too much! See output above.'
    return idealized
