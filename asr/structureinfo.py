from asr.core import command


def get_reduced_formula(formula, stoichiometry=False):
    """
    Returns the reduced formula corresponding to a chemical formula,
    in the same order as the original formula
    E.g. Cu2S4 -> CuS2

    Parameters:
        formula (str)
        stoichiometry (bool): if True, return the stoichiometry ignoring the
          elements appearing in the formula, so for example "AB2" rather than
          "MoS2"
    Returns:
        A string containing the reduced formula
    """
    from functools import reduce
    from fractions import gcd
    import string
    import re
    split = re.findall('[A-Z][^A-Z]*', formula)
    matches = [re.match('([^0-9]*)([0-9]+)', x)
               for x in split]
    numbers = [int(x.group(2)) if x else 1 for x in matches]
    symbols = [matches[i].group(1) if matches[i] else split[i]
               for i in range(len(matches))]
    divisor = reduce(gcd, numbers)
    result = ''
    numbers = [x // divisor for x in numbers]
    numbers = [str(x) if x != 1 else '' for x in numbers]
    if stoichiometry:
        numbers = sorted(numbers)
        symbols = string.ascii_uppercase
    for symbol, number in zip(symbols, numbers):
        result += symbol + number
    return result


def has_inversion(atoms, use_spglib=True):
    """
    Parameters:
        atoms: Atoms object
            atoms
        use_spglib: bool
            use spglib
    Returns:
        out: bool
    """
    import numpy as np
    try:
        import spglib
    except ImportError as x:
        import warnings
        warnings.warn('using gpaw symmetry for inversion instead: {}'
                      .format(x))
        use_spglib = False

    atoms2 = atoms.copy()
    atoms2.pbc[:] = True
    atoms2.center(axis=2)
    if use_spglib:
        R = -np.identity(3, dtype=int)
        r_n = spglib.get_symmetry(atoms2, symprec=1.0e-3)['rotations']
        return np.any([np.all(r == R) for r in r_n])
    else:
        from gpaw.symmetry import atoms2symmetry
        return atoms2symmetry(atoms2).has_inversion


def webpanel(row, key_descriptions):
    from ase.db.summary import ATOMS, UNITCELL
    from asr.browser import table

    basictable = table(row, 'Structural info', [
        'prototype', 'class', 'spacegroup', 'gap', 'magstate', 'ICSD_id',
        'COD_id'
    ], key_descriptions, 2)
    rows = basictable['rows']
    codid = row.get('COD_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/' +
                    '{id}.html">{id}</a>'.format(id=codid))
            if 'COD' in tmprow[0]:
                tmprow[1] = href

    doi = row.get('doi')
    if doi:
        rows.append([
            'Monolayer DOI',
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    panel = {'title': 'Summary',
             'columns': [[UNITCELL, basictable], [ATOMS]],
             'sort': 1}
    return [panel]


tests = [{'description': 'Test SI.',
          'cli': ['asr run "setup.materials -s Si2"',
                  'ase convert materials.json structure.json',
                  'asr run "setup.params asr.gs@calculate:ecut 300 '
                  'asr.gs@calculate:kptdensity 2"',
                  'asr run structureinfo',
                  'asr run database.fromtree',
                  'asr run "browser --only-figures"']}]


@command('asr.structureinfo',
         tests=tests,
         requires=['structure.json'],
         webpanel=webpanel)
def main():
    """Get structural information of atomic structure.

    This recipe produces information such as the space group and magnetic
    state properties that requires only an atomic structure. This recipes read
    the atomic structure in `structure.json`.
    """

    import numpy as np
    from ase.io import read
    from pathlib import Path

    atoms = read('structure.json')
    info = {}

    folder = Path().cwd()
    info['folder'] = str(folder)

    # Determine magnetic state
    def get_magstate(a):
        magmom = a.get_magnetic_moment()
        if abs(magmom) > 0.02:
            return 'fm'

        magmoms = a.get_magnetic_moments()
        if abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
            return 'afm'

        # Material is essentially non-magnetic
        return 'nm'

    try:
        magstate = get_magstate(atoms)
    except RuntimeError:
        magstate = 'nm'
    info['magstate'] = magstate.upper()

    formula = atoms.get_chemical_formula(mode='metal')
    stoichimetry = get_reduced_formula(formula, stoichiometry=True)
    info['formula'] = formula
    info['stoichiometry'] = stoichimetry
    info['has_inversion_symmetry'] = has_inversion(atoms)

    def coarsesymmetries(a):
        from gpaw.symmetry import Symmetry
        cell_cv = a.get_cell()
        tol = 0.01  # Tolerance for coarse symmetries
        coarsesymmetry = Symmetry(
            a.get_atomic_numbers(),
            cell_cv,
            tolerance=tol,
            symmorphic=False,
            time_reversal=True)
        coarsesymmetry.analyze(a.get_scaled_positions())
        return (coarsesymmetry.op_scc, coarsesymmetry.ft_sc)

    op_scc, ft_sc = coarsesymmetries(atoms)
    symmetry = [(op_cc.tolist(), ft_c.tolist())
                for op_cc, ft_c in zip(op_scc, ft_sc)]
    info['symmetries'] = symmetry
    import spglib
    sg, number = spglib.get_spacegroup(atoms, symprec=1e-4).split()
    number = int(number[1:-1])
    info['spacegroup'] = sg
    info['spgnum'] = number

    # Set temporary uid.
    # Will be changed later once we know the prototype.
    info['is_magnetic'] = info['magstate'] != 'NM'

    if (atoms.pbc == [True, True, False]).all():
        info['cell_area'] = abs(np.linalg.det(atoms.cell[:2, :2]))

    info['__key_descriptions__'] = {
        'magstate': 'KVP: Magnetic state',
        'is_magnetic': 'KVP: Material is magnetic (Magnetic)',
        'cell_area': 'KVP: Area of unit-cell [Ang^2]',
        'has_invsymm': 'KVP: Inversion symmetry',
        'stoichiometry': 'KVP: Stoichiometry',
        'spacegroup': 'KVP: Space group',
        'spgnum': 'KVP: Space group number',
        'prototype': 'KVP: Prototype'}

    return info


if __name__ == '__main__':
    main.cli()
