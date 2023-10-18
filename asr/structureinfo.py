"""Structural information."""
import numpy as np

from asr.core import command
from asr.paneldata import StructureInfoResult


def get_reduced_formula(formula, stoichiometry=False):
    """Get reduced formula from formula.

    Returns the reduced formula corresponding to a chemical formula,
    in the same order as the original formula
    E.g. Cu2S4 -> CuS2

    Parameters
    ----------
    formula : str
    stoichiometry : bool
        If True, return the stoichiometry ignoring the
        elements appearing in the formula, so for example "AB2" rather than
        "MoS2"

    Returns
    -------
        A string containing the reduced formula.
    """
    import re
    import string
    from functools import reduce
    from math import gcd
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


tests = [{'description': 'Test SI.',
          'cli': ['asr run "setup.materials -s Si2"',
                  'ase convert materials.json structure.json',
                  'asr run "setup.params asr.gs@calculate:ecut 300 '
                  'asr.gs@calculate:kptdensity 2"',
                  'asr run structureinfo',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


def get_layer_group(atoms, symprec):
    try:
        from spglib.spglib import get_symmetry_layerdataset
    except ImportError:
        return None, None

    assert atoms.pbc.sum() == 2
    aperiodic_dir = np.where(~atoms.pbc)[0][0]
    # Prepare for spglib v3 API change to always have the aperiodic_dir == 2
    # See: https://github.com/spglib/spglib/issues/314.
    if aperiodic_dir != 2:
        perm = np.array([0, 1, 2])
        # Swap axes such that aperiodic is always 2
        perm[2], perm[aperiodic_dir] = perm[aperiodic_dir], perm[2]
        atoms = atoms.copy()
        atoms.set_pbc(atoms.get_pbc()[perm])
        # The atoms are stored in cartesian coordinates, therefore, we are
        # free to permute the cell vectors and system remains invariant.
        atoms.set_cell(atoms.get_cell()[perm], scale_atoms=False)
        aperiodic_dir = 2

    assert aperiodic_dir == 2

    lg_dct = get_symmetry_layerdataset(
        (atoms.get_cell(),
         atoms.get_scaled_positions(),
         atoms.get_atomic_numbers()),
        symprec=symprec,
        aperiodic_dir=aperiodic_dir)

    layergroup = lg_dct['number']
    layergroupname = lg_dct['international']

    return layergroupname, layergroup


@command('asr.structureinfo',
         tests=tests,
         requires=['structure.json'],
         returns=StructureInfoResult)
def main() -> StructureInfoResult:
    """Get structural information of atomic structure.

    This recipe produces information such as the space group and magnetic
    state properties that requires only an atomic structure. This recipes read
    the atomic structure in `structure.json`.
    """
    import numpy as np
    from ase.io import read

    from asr.utils.symmetry import c2db_symmetry_angle, c2db_symmetry_eps

    atoms = read('structure.json')
    info = {}

    formula = atoms.get_chemical_formula(mode='metal')
    stoichimetry = get_reduced_formula(formula, stoichiometry=True)
    info['formula'] = formula
    info['stoichiometry'] = stoichimetry

    # Get crystal symmetries
    from asr.utils.symmetry import atoms2symmetry

    symmetry = atoms2symmetry(atoms,
                              tolerance=c2db_symmetry_eps,
                              angle_tolerance=c2db_symmetry_angle)
    info['has_inversion_symmetry'] = symmetry.has_inversion
    dataset = symmetry.dataset
    info['spglib_dataset'] = dataset

    # Get crystal type
    stoi = atoms.symbols.formula.stoichiometry()[0]
    sg = dataset['international']
    number = dataset['number']
    pg = dataset['pointgroup']
    w = ''.join(sorted(set(dataset['wyckoffs'])))
    crystal_type = f'{stoi}-{number}-{w}'

    info['layergroup'] = None
    info['lgnum'] = None
    ndims = sum(atoms.pbc)
    if ndims == 2:
        info['layergroup'], info['lgnum'] = get_layer_group(
            atoms,
            symprec=c2db_symmetry_eps)

    info['crystal_type'] = crystal_type
    info['spacegroup'] = sg
    info['spgnum'] = number
    from ase.db.core import convert_str_to_int_float_or_str, str_represents
    if str_represents(pg):
        info['pointgroup'] = convert_str_to_int_float_or_str(pg)
    else:
        info['pointgroup'] = pg

    if (atoms.pbc == [True, True, False]).all():
        info['cell_area'] = abs(np.linalg.det(atoms.cell[:2, :2]))
    else:
        info['cell_area'] = None

    return StructureInfoResult(data=info)


if __name__ == '__main__':
    main.cli()
