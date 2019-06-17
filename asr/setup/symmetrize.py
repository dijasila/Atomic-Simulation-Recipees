from asr.utils import command, option

@command('asr.setup.symmetrize',
         save_results_file=False)
@option('--tolerance', type=float, default=1e-3,
        help='Tolerance when evaluating symmetries')
def main(tolerance):
    """Symmetrize atomic structure.

    This function changes the atomic positions and the unit cell
    of an approximately symmetrical structure into an exactly
    symmetrical structure.

    In practice, the spacegroup of the structure located in 'original.json'
    is evaluated using a not-very-strict tolerance, which can be adjusted using
    the --tolerance switch. Then the symmetries of the spacegroup are used
    to generate equivalent atomic structures and by taking an average of these
    atomic positions we generate an exactly symmetric atomic structure.

    \b
    Examples:
    ---------
    Symmetrize an atomic structure using the default tolerance
        asr run setup.symmetrize
    """

    import numpy as np
    import spglib
    from ase.io import read, write
    atoms = read('original.json')

    spos_ac = atoms.get_scaled_positions(wrap=False)
    cell_cv = atoms.get_cell()
    
    symmetry = spglib.get_symmetry(atoms, symprec=tolerance)
    spacegroup = spglib.get_spacegroup(atoms, symprec=tolerance)

    uspos_sac = []
    M_scc = []
    for U_cc, t_c in zip(symmetry['rotations'], symmetry['translations']):
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
    spos_ac = np.mean(uspos_sac, axis=0)
    M_cc = np.mean(M_scc, axis=0)

    from ase.geometry.cell import cellpar_to_cell
    dotprods = M_cc[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    l_c = np.sqrt(dotprods[:3])
    angles_c = np.arccos(dotprods[3:] / l_c[[1, 0, 0]] / l_c[[2, 2, 1]])
    angles_c *= 180 / np.pi

    cp = np.concatenate([l_c, angles_c])
    origcp = atoms.cell.cellpar()

    nsym = len(symmetry['rotations'])
    print(f'Forcing structure into spacegroup {spacegroup} '
          f'with {nsym} symmetries (tol: {tolerance})')
    
    a1, b1, c1, alpha1, beta1, gamma1 = cp - origcp

    print('Cell Change: (Δa, Δb, Δc, Δα, Δβ, Δγ) = '
          f'({a1:.1e} Å, {b1:.1e} Å, {c1:.1e} Å, '
          f'{alpha1:.2f}°, {beta1:.2f}°, {gamma1:.2f}°)')

    origcell = atoms.get_cell()
    ab_normal = np.cross(origcell[0], origcell[1])
    cell = cellpar_to_cell(cp, ab_normal=ab_normal,
                           a_direction=origcell[0])

    origspos_ac = atoms.get_scaled_positions(wrap=False)
    dpos_av = np.dot(spos_ac - origspos_ac, cell)
    dpos_a = np.sqrt(np.sum(dpos_av**2, 1))
    with np.printoptions(precision=2, suppress=False):
        print(f'Change of pos.:')
        msg = '    '
        for symbol, dpos in zip(atoms.symbols, dpos_a):
            msg += f' {symbol}: {dpos:.1e} Å,'
            if len(msg) > 70:
                print(msg[:-1])
                msg = '    '
        print(msg[:-1])
    atoms.set_cell(cell)
    atoms.set_scaled_positions(spos_ac)

    # Sanity check
    newspacegroup = spglib.get_spacegroup(atoms, symprec=tolerance)
    assert spacegroup == newspacegroup
    write('unrelaxed.json', atoms)


group = 'setup'


if __name__ == '__main__':
    main()
