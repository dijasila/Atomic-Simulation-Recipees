from asr.core import command, option

@command(module='asr.duplicates',
         requires=['structure.json'],
         resources='1:20m')


def check_duplicates(structure, db):
    from ase.formula import Formula

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
            struc = db.get_atoms(id=id)
            if row.get('magstate') == ref_mag:
                struc = asetopy.get_structure(struc)
                structure_list.append(struc)
    rmdup = RemoveExistingFilter(structure_list, matcher,symprec=1e-5)
    results = rmdup.test(refpy)
    print('INFO: structure already in DB? {}'.format(not results))

    return not results


def main():
    """
    tbd
    """
    from ase.db import connect
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter
    from asr.core import read_json
    from ase.io import write, read

    startset = connect('db.db')

    asetopy = AseAtomsAdaptor()

    structure = read('structure.json')
    struc_info = read_json('results-asr.structureinfo.json')
    ref_mag = struc_info.get('magstate')

    refpy = asetopy.get_structure(structure)
    matcher = StructureMatcher()

    does_exist = check_duplicates(structure, startset)

    results = {'duplicate': does_exist,
               '__key_descriptions__': {'duplicate':
               'Does a duplicate of structure.json already exist in the DB?'}}

    return results
