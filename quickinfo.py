def stoichiometry(kvp, data, atoms, verbose):
    formula = atoms.get_chemical_formula()
    kvp['stoichiometry'] = get_reduced_formula(formula,
                                               stoichiometry=True)


def get_started(kvp, data, skip_forces):
    folder, state = Path().cwd().parts[-2:]
    assert state in {'nm', 'fm', 'afm'}, state
    formula, _, _ = folder.partition('-')
    e_nm = read('../relax-nm.traj').get_potential_energy()
    if os.path.isfile('gs.gpw'):
        atoms = read('gs.gpw')
        calc = atoms.calc
    else:
        atoms = read('../relax-{}.traj'.format(state))
        calc = None

    for repeat in [1, 2]:
        formula = Atoms(formula * repeat).get_chemical_formula()
        if formula == atoms.get_chemical_formula():
            break  # OK
    else:
        raise ValueError('Wrong folder name: ' + folder)

    f = atoms.get_forces()
    s = atoms.get_stress()[:2]
    fmax = ((f**2).sum(1).max())**0.5
    smax = abs(s).max()

    # Allow for a bit of slack because of a small bug in our
    # modified BFGS:
    slack = 0.002
    if len(atoms) < 50 and not skip_forces:
        assert fmax < 0.01 + slack, fmax
        assert smax < 0.002, smax
    kvp['smaxinplane'] = smax

    if state == 'nm':
        assert not atoms.calc.get_spin_polarized()
        atoms.calc.results['magmom'] = 0.0
    else:
        if calc is not None:
            assert atoms.calc.get_spin_polarized()
        m = atoms.get_magnetic_moment()
        ms = atoms.get_magnetic_moments()
        if state == 'fm':
            assert abs(m) > 0.1
        else:  # afm
            assert abs(m) < 0.02 and abs(ms).max() > 0.1
    kvp['magstate'] = state.upper()
    kvp['is_magnetic'] = state != 'nm'
    kvp['cell_area'] = np.linalg.det(atoms.cell[:2, :2])
    kvp['has_invsymm'] = has_inversion(atoms)
    if state != 'nm':
        kvp['dE_NM'] = 1000 * ((atoms.get_potential_energy() - e_nm) /
                               len(atoms))
    # trying to make small negative numbers positive
    # cell = atoms.cell
    # atoms.cell = np.where(abs(cell) < 1.0e-14, 0.0, cell)
    return atoms, folder, state


def spacegroup(kvp, data, atoms, verbose):
    try:
        import spglib
    except ImportError:
        pass
    else:
        sg, number = spglib.get_spacegroup(atoms, symprec=1e-4).split()
        number = int(number[1:-1])
        print('Spacegroup:', sg, number)
        kvp['spacegroup'] = sg
