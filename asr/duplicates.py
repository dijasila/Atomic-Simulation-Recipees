from asr.core import command, argument


def check_duplicates(structure=None, row=None, db=None,
                     comparison_keys=[],
                     exclude_ids=set(),
                     extra_data={},
                     verbose=False,
                     symprec=1e-5):
    """Compare structure with structures in db with magstate ref_mag."""
    from ase.formula import Formula
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter

    if row is not None:
        assert structure is None
        structure = row.toatoms()
        extra_row_data = row
    else:
        assert structure is not None
        extra_row_data = extra_data

    asetopy = AseAtomsAdaptor()
    refpy = asetopy.get_structure(structure)
    matcher = StructureMatcher()

    symbols = set(structure.get_chemical_symbols())
    formula = Formula(str(structure.symbols))
    stoichiometry = formula.reduce()[0]
    id_duplicates = []

    # Stoichiometric identification
    for dbrow in db.select(','.join(symbols), include_data=False):
        if dbrow.id in exclude_ids:
            continue
        stoichiometry_row = Formula(str(dbrow.get("formula"))).reduce()[0]
        if stoichiometry_row == stoichiometry:
            id = dbrow.get("id")
            struc = dbrow.toatoms()
            if all(dbrow.get(key) == extra_row_data.get(key)
                   for key in comparison_keys):
                struc = asetopy.get_structure(struc)
                rmdup = RemoveExistingFilter([struc],
                                             matcher,
                                             symprec=symprec)
                is_duplicate = not rmdup.test(refpy)
                if is_duplicate:
                    id_duplicates.append(id)

    has_duplicate = bool(len(id_duplicates))
    if verbose:
        print('INFO: structure already in DB? {}'.format(has_duplicate))

    return has_duplicate, id_duplicates


@command(module='asr.duplicates',
         requires=['structure.json', 'results-asr.structureinfo.json'],
         resources='1:20m')
@argument('database')
def main(database):
    """Identify duplicates of structure.json in given database.

    This recipe reads in a structure.json and identifies duplicates of
    that structure in an existing DB db.db. It uses the
    StructureMatcher object from pymatgen
    (https://pymatgen.org/pymatgen.analysis.structure_matcher.html). This
    is done by reducing the structures to their respective primitive
    cells and uses the normalized average rms displacement to evaluate
    the similarity of two structures.

    """
    from ase.db import connect
    from asr.core import read_json
    from ase.io import read

    startset = connect(database)
    structure = read('structure.json')
    struc_info = read_json('results-asr.structureinfo.json')
    ref_mag = struc_info.get('magstate')

    does_exist, id_list = check_duplicates(structure, startset, ref_mag)

    print('INFO: duplicate structures in db: {}'.format(id_list))

    results = {'duplicate': does_exist,
               'duplicate_IDs': id_list,
               '__key_descriptions__':
               {'duplicate':
                'Does a duplicate of structure.json already exist in the DB?',
                'duplicate_IDs':
                'list of IDs of identified duplicates in original DB'}}

    return results
