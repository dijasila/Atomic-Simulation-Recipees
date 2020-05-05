from asr.core import command, argument


def check_duplicates(structure, db, ref_mag=None,
                     exclude_ids=set(), verbose=False):
    """Compare structure with structures in db with magstate ref_mag."""
    from ase.formula import Formula
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter

    asetopy = AseAtomsAdaptor()
    refpy = asetopy.get_structure(structure)
    matcher = StructureMatcher()

    symbols = set(structure.get_chemical_symbols())
    formula = Formula(str(structure.symbols))
    stoichiometry = formula.reduce()[0]
    id_duplicates = []

    # Stoichiometric identification
    for row in db.select(','.join(symbols), include_data=False):
        if row.id in exclude_ids:
            continue
        stoichiometry_row = Formula(str(row.get("formula"))).reduce()[0]
        if stoichiometry_row == stoichiometry:
            id = row.get("id")
            struc = row.toatoms()
            if row.get('magstate') == ref_mag:
                struc = asetopy.get_structure(struc)
                rmdup = RemoveExistingFilter([struc],
                                             matcher,
                                             symprec=1e-5)
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
