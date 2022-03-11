"""Structural information."""
import numpy as np
from ase import Atoms

from asr.core import command, ASRResult, prepare_result, option, AtomsFile


def get_spg_href(url):
    from asr.database.browser import href

    return href('SpgLib', url)


def describe_pointgroup_entry(spglib):
    from asr.database.browser import describe_entry

    pointgroup = describe_entry(
        'pointgroup',
        f"Point group determined with {spglib}."
    )

    return pointgroup


def webpanel(result, context):
    from asr.database.browser import (table, describe_entry, code, bold,
                                      br, href, dl, div)

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    crystal_type = describe_entry(
        'crystal_type',
        "The crystal type is defined as "
        + br
        + div(bold('-'.join([code('stoi'),
                             code('spg no.'),
                             code('occ. wyck. pos.')])), 'well well-sm text-center')
        + 'where'
        + dl(
            [
                [code('stoi'), 'Stoichiometry.'],
                [code('spg no.'), f'The space group calculated with {spglib}.'],
                [code('occ. wyck. pos.'),
                 'Alphabetically sorted list of occupied '
                 f'wyckoff positions determined with {spglib}.'],
            ]
        )
    )

    cls = describe_entry(
        'class',
        "The material class is a manually attributed name that is given to "
        "a material for historical reasons and is therefore not well-defined "
        "but can be useful classifying materials."
    )

    spg_list_link = href(
        'Space group', 'https://en.wikipedia.org/wiki/List_of_space_groups'
    )
    spacegroup = describe_entry(
        'spacegroup',
        f"{spg_list_link} determined with {spglib}."
    )

    spgnum = describe_entry(
        'spgnum',
        f"{spg_list_link} number determined with {spglib}."
    )

    pointgroup = describe_pointgroup_entry(spglib)

    # XXX We should define a different panel for these external DB IDs
    # and DOI.  Those are not related to this recipe!

    # icsd_link = href('Inorganic Crystal Structure Database (ICSD)',
    #                  'https://icsd.products.fiz-karlsruhe.de/')

    # icsd_id = describe_entry(
    #     'icsd_id',
    #     f"ID of a closely related material in the {icsd_link}."
    # )

    # cod_link = href(
    #     'Crystallography Open Database (COD)',
    #     'http://crystallography.net/cod/browse.html'
    # )

    # cod_id = describe_entry(
    #     'cod_id',
    #     f"ID of a closely related material in the {cod_link}."
    # )

    basictable = table(result, 'Structure info', [
        crystal_type, cls, spacegroup, spgnum, pointgroup,
        # icsd_id, cod_id
    ], context.descriptions, 2)
    basictable['columnwidth'] = 4
    # rows = basictable['rows']

    # codid = row.get('cod_id')
    # if codid:
    #     # Monkey patch to make a link
    #     for tmprow in rows:
    #         href = ('<a href="http://www.crystallography.net/cod/'
    #                 + '{id}.html">{id}</a>'.format(id=codid))
    #         if 'cod_id' in tmprow[0]:
    #             tmprow[1] = href

    # doi = row.get('doi')
    # doistring = describe_entry(
    #     'Reported DOI',
    #     'DOI of article reporting the synthesis of the material.'
    # )
    # if doi:
    #     rows.append([
    #         doistring,
    #         '<a href="https://doi.org/{doi}" target="_blank">{doi}'
    #         '</a>'.format(doi=doi)
    #    ])

    panel = {'title': 'Summary',
             'columns': [[basictable,
                          {'type': 'table', 'header': ['Stability', ''],
                           'rows': [],
                           'columnwidth': 4}],
                         [{'type': 'atoms'}, {'type': 'cell'}]],
             'sort': -1}
    return [panel]


@prepare_result
class Result(ASRResult):

    cell_area: float
    has_inversion_symmetry: bool
    stoichiometry: str
    spacegroup: str
    spgnum: int
    pointgroup: str
    crystal_type: str
    spglib_dataset: dict
    formula: str

    key_descriptions = {
        "cell_area": "Area of unit-cell [`Å²`]",
        "has_inversion_symmetry": "Material has inversion symmetry",
        "stoichiometry": "Stoichiometry",
        "spacegroup": "Space group",
        "spgnum": "Space group number",
        "pointgroup": "Point group",
        "crystal_type": "Crystal type",
        "spglib_dataset": "SPGLib symmetry dataset.",
        "formula": "Chemical formula."
    }

    formats = {"webpanel2": webpanel}


@command('asr.structureinfo')
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
def main(atoms: Atoms) -> Result:
    """Get structural information of atomic structure.

    This recipe produces information such as the space group and magnetic
    state properties that requires only an atomic structure. This recipes read
    the atomic structure in `structure.json`.
    """
    from asr.utils.symmetry import c2db_symmetry_eps

    info = {}

    formula = atoms.symbols.formula.convert('metal')
    info['formula'] = str(formula)
    info['stoichiometry'] = formula.stoichiometry()[0]

    # Get crystal symmetries
    from asr.utils.symmetry import atoms2symmetry
    # According to tests by Thomas Olsen on C2DB, having a coarse
    # angle tolerance is not important for solving the issue documented
    # for asr.utils.symmetry.c2db_symmetry_eps.  So we still use a very
    # strict symmetry.
    symmetry = atoms2symmetry(atoms,
                              tolerance=c2db_symmetry_eps,
                              angle_tolerance=0.1)
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
    info['crystal_type'] = crystal_type
    info['spacegroup'] = sg
    info['spgnum'] = number
    from ase.db.core import str_represents, convert_str_to_int_float_or_str
    if str_represents(pg):
        info['pointgroup'] = convert_str_to_int_float_or_str(pg)
    else:
        info['pointgroup'] = pg

    if (atoms.pbc == [True, True, False]).all():
        info['cell_area'] = abs(np.linalg.det(atoms.cell[:2, :2]))
    else:
        info['cell_area'] = None

    return Result(data=info)
