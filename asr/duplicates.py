from asr.core import command, argument


def get_rmsd(atoms1, atoms2, adaptor=None, matcher=None):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    if adaptor is None:
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

    if matcher is None:
        matcher = StructureMatcher()

    struct1 = adaptor.get_structure(atoms1)
    struct2 = adaptor.get_structure(atoms2)

    struct1, struct2 = matcher._process_species([struct1, struct2])

    if not matcher._subset and matcher._comparator.get_hash(struct1.composition) \
            != matcher._comparator.get_hash(struct2.composition):
        return None

    struct1, struct2, fu, s1_supercell = matcher._preprocess(struct1, struct2)
    match = matcher._match(struct1, struct2, fu, s1_supercell,
                           break_on_match=True)
    if match is None:
        return False
    else:
        return match[0]


def fix_nonpbc_cell_directions(atoms):
    import numpy as np
    atoms = atoms.copy()
    pbc_c = atoms.get_pbc()
    cell_cv = atoms.get_cell()
    print('before cell', cell_cv)
    cell_cv[~pbc_c] = np.eye(3)[~pbc_c]
    print('after cell', cell_cv)
    atoms.set_cell(cell_cv)
    return atoms


def are_structures_duplicates(atoms1, atoms2, symprec=1e-5,
                              adaptor=None, matcher=None):
    """Return true if atoms1 and atoms2 are duplicates."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter

    if adaptor is None:
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

    if matcher is None:
        matcher = StructureMatcher()

    atoms1 = fix_nonpbc_cell_directions(atoms1)
    atoms2 = fix_nonpbc_cell_directions(atoms2)
    struc1 = adaptor.get_structure(atoms1)
    struc2 = adaptor.get_structure(atoms2)
    rmdup = RemoveExistingFilter([struc2],
                                 matcher,
                                 symprec=symprec)
    rmsd = get_rmsd(atoms1, atoms2, matcher=matcher, adaptor=adaptor)
    print('rmsd', rmsd)
    is_duplicate = not rmdup.test(struc1)
    return is_duplicate


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

    if row is not None:
        assert structure is None
        structure = row.toatoms()
        extra_row_data = row
    else:
        assert structure is not None
        extra_row_data = extra_data

    adaptor = AseAtomsAdaptor()
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
            otherstructure = dbrow.toatoms()
            if all(dbrow.get(key) == extra_row_data.get(key)
                   for key in comparison_keys):
                is_duplicate = are_structures_duplicates(structure,
                                                         otherstructure,
                                                         matcher=matcher,
                                                         adaptor=adaptor,
                                                         symprec=symprec)
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
