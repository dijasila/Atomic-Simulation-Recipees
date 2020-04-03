from asr.core import command


def check_duplicates(structure, db, ref_mag):
    from ase.formula import Formula
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter

    asetopy = AseAtomsAdaptor()
    refpy = asetopy.get_structure(structure)
    matcher = StructureMatcher()

    formula = Formula(str(structure.symbols))
    stoichiometry = formula.reduce()[0]
    id_duplicates = []
    structure_list = []
    # Stoichiometric identification
    for row in db.select():
        stoichiometry_row = Formula(str(row.get("formula"))).reduce()[0]
        if stoichiometry_row == stoichiometry:
            id = row.get("id")
            id_duplicates.append(id)
            struc = row.toatoms()
            if row.get('magstate') == ref_mag:
                struc = asetopy.get_structure(struc)
                structure_list.append(struc)

    rmdup = RemoveExistingFilter(structure_list, matcher, symprec=1e-5)
    results = rmdup.test(refpy)
    print('INFO: structure already in DB? {}'.format(not results))

    return not results, id_duplicates


@command(module='asr.duplicates',
         requires=['structure.json'],
         resources='1:20m')
def main():
    """
    tbd
    """
    from ase.db import connect
    from asr.core import read_json
    from ase.io import read

    startset = connect('db.db')
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
                'Returns a list of row IDs corresponding to the IDs of the duplicates in the original DB'}}

    return results
