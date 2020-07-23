from asr.core import command


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
    from functools import reduce
    from math import gcd
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


def webpanel(row, key_descriptions):
    from asr.database.browser import table

    basictable = table(row, 'Structure info', [
        'crystal_prototype', 'class', 'spacegroup', 'spgnum', 'ICSD_id',
        'COD_id'
    ], key_descriptions, 2)
    basictable['columnwidth'] = 4
    rows = basictable['rows']
    codid = row.get('COD_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/'
                    + '{id}.html">{id}</a>'.format(id=codid))
            if 'COD' in tmprow[0]:
                tmprow[1] = href

    doi = row.get('doi')
    if doi:
        rows.append([
            'Monolayer reported DOI',
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    panel = {'title': 'Summary',
             'columns': [[basictable,
                          {'type': 'table', 'header': ['Stability', ''],
                           'rows': [],
                           'columnwidth': 4}],
                         [{'type': 'atoms'}, {'type': 'cell'}]],
             'sort': -1}
    return [panel]


tests = [{'description': 'Test SI.',
          'cli': ['asr run "setup.materials -s Si2"',
                  'ase convert materials.json structure.json',
                  'asr run "setup.params asr.gs@calculate:ecut 300 '
                  'asr.gs@calculate:kptdensity 2"',
                  'asr run structureinfo',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


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

    atoms = read('structure.json')
    info = {}

    formula = atoms.get_chemical_formula(mode='metal')
    stoichimetry = get_reduced_formula(formula, stoichiometry=True)
    info['formula'] = formula
    info['stoichiometry'] = stoichimetry

    # Get crystal symmetries
    from asr.utils.symmetry import atoms2symmetry
    symmetry = atoms2symmetry(atoms,
                              tolerance=1e-3,
                              angle_tolerance=0.1)
    info['has_inversion_symmetry'] = symmetry.has_inversion
    dataset = symmetry.dataset
    info['spglib_dataset'] = dataset

    # Get crystal prototype
    stoi = atoms.symbols.formula.stoichiometry()[0]
    sg = dataset['international']
    number = dataset['number']
    w = ''.join(sorted(set(dataset['wyckoffs'])))
    crystal_prototype = f'{stoi}-{number}-{w}'
    info['crystal_prototype'] = crystal_prototype
    info['spacegroup'] = sg
    info['spgnum'] = number

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
        'crystal_prototype': 'KVP: Crystal prototype'}

    return info


if __name__ == '__main__':
    main.cli()
