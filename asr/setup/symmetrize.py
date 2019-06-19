from asr.utils import command, option


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
                    line += f' + ({ft:.3e})'
            lines.append(line)
        lines.append('')
    return '\n'.join(lines)


def atomstospgcell(atoms, magmoms=None):
    from ase.calculators.calculator import PropertyNotImplementedError
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions(wrap=False)
    numbers = atoms.get_atomic_numbers()
    if magmoms is None:
        try:
            magmoms = atoms.get_magnetic_moments()
        except (RuntimeError, PropertyNotImplementedError):
            magmoms = None
    if magmoms is not None:
        return (lattice, positions, numbers, magmoms)
    else:
        return (lattice, positions, numbers)


def symmetrize_atoms(atoms, tolerance=None, angle_tolerance=None,
                     return_dataset=False):
    """ Atoms() -> SymmetrizedAtoms()"""

    import spglib
    import numpy as np
    from ase import Atoms

    spgcell = atomstospgcell(atoms)
    if len(spgcell) > 3:
        magmoms_a = spgcell[3]
    else:
        magmoms_a = None
        newmagmoms_a = None
    dataset = spglib.get_symmetry_dataset(spgcell,
                                          symprec=tolerance,
                                          angle_tolerance=angle_tolerance)
    
    newspgcell = spglib.standardize_cell(spgcell, symprec=tolerance,
                                         to_primitive=True,
                                         no_idealize=False,
                                         angle_tolerance=angle_tolerance)
    newspgcell = spglib.standardize_cell(newspgcell, symprec=tolerance,
                                         to_primitive=True,
                                         no_idealize=False,
                                         angle_tolerance=angle_tolerance)
    if magmoms_a:
        newmagmoms_a = np.zeros(len(newspgcell[1]), float)
        newmagmoms_a[dataset['mapping_to_primitive']] = magmoms_a

    newdataset = spglib.get_symmetry_dataset(newspgcell,
                                             symprec=tolerance,
                                             angle_tolerance=angle_tolerance)

    cell = newspgcell[0]
    spos_ac = newspgcell[1]
    numbers = newspgcell[2]
    idealized = Atoms(numbers=numbers, scaled_positions=spos_ac, cell=cell,
                      pbc=True, magmoms=newmagmoms_a)
    if return_dataset:
        return idealized, dataset, newdataset

    return idealized
    

@command('asr.setup.symmetrize',
         save_results_file=False)
@option('--tolerance', type=float, default=1e-2,
        help='Tolerance when evaluating symmetries')
@option('--angle-tolerance', type=float, default=0.1,
        help='Tolerance one angles when evaluating symmetries')
def main(tolerance, angle_tolerance):
    """Symmetrize atomic structure.

    This function changes the atomic positions and the unit cell
    of an approximately symmetrical structure into an exactly
    symmetrical structure.

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
        idealized, dataset1, dataset2 = symmetrize_atoms(idealized,
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

    nsym = len(dataset2['rotations'])

    # Sanity check
    from gpaw.symmetry import Symmetry
    id_a = idealized.get_atomic_numbers()
    symmetry = Symmetry(id_a, idealized.cell, idealized.pbc,
                        symmorphic=False,
                        time_reversal=False)
    symmetry.analyze(idealized.get_scaled_positions())

    nsymgpaw = len(symmetry.op_scc)

    write('unrelaxed.json', idealized)

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
    main()
