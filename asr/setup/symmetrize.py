from asr.utils import command, option

@command('asr.setup.symmetrize',
         save_results_file=False)
@option('--tolerance', type=float, default=1e-3,
        help='Tolerance when evaluating symmetries')
def main(tolerance):
    """Symmetrize atomic structure.

    

    \b
    Examples:
    ---------
    Set up all known magnetic configurations (assuming existence of
    'unrelaxed.json')
        asr run setup.magnetize
    \b
    Only set up ferromagnetic configuration
        asr run setup.magnetic --state fm
    """

    import numpy as np
    import spglib
    from ase.io import read, write
    atoms = read('original.json')

    spos_ac = atoms.get_scaled_positions()
    spos_ac -= spos_ac[0] * atoms.pbc
    atoms.set_scaled_positions(spos_ac)
    spos_ac = atoms.get_scaled_positions()
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
    print(f'Spacegroup {spacegroup} with {nsym} symmetries (tol: {tolerance})')
    
    # a0, b0, c0, alpha0, beta0, gamma0 = origcp
    # a, b, c, alpha, beta, gamma = cp
    a1, b1, c1, alpha1, beta1, gamma1 = cp - origcp

    print('Cell Change: (Δa, Δb, Δc, Δα, Δβ, Δγ) = '
          f'({a1:.5f}, {b1:.5f}, {c1:.5f}, '
          f'{alpha1:.5f}°, {beta1:.5f}°, {gamma1:.5f}°)')

    cell = cellpar_to_cell(cp)
    print(cell - atoms.get_cell())
    
    origspos_ac = atoms.get_scaled_positions()
    dspos_ac = spos_ac - origspos_ac
    with np.printoptions(precision=2, suppress=False):
        print(f'Change of (scaled) pos.:')
        print(dspos_ac)
    atoms.set_cell(cell)
    atoms.set_scaled_positions(spos_ac)

    # Sanity check
    newspacegroup = spglib.get_spacegroup(atoms, symprec=tolerance)
    assert spacegroup == newspacegroup
    write('unrelaxed.json', atoms)


group = 'setup'


if __name__ == '__main__':
    main()
