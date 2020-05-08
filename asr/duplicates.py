from asr.core import command, argument
import numpy as np


def get_rmsd(atoms1, atoms2, adaptor=None, matcher=None):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from ase.build import niggli_reduce

    if adaptor is None:
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

    if matcher is None:
        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)

    pbc_c = atoms1.get_pbc()
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.set_pbc(True)
    atoms2.set_pbc(True)
    niggli_reduce(atoms1)
    niggli_reduce(atoms2)
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
        return None
    else:
        rmsd = match[0]
        # Fix normalization
        vol = atoms1.get_volume()
        natoms = len(atoms1)
        old_norm = (natoms / vol)**(1 / 3)
        rmsd /= old_norm  # Undo
        lenareavol = np.linalg.det(atoms1.get_cell()[pbc_c][:, pbc_c])
        new_norm = (natoms / lenareavol)**(1 / sum(pbc_c))
        rmsd *= new_norm  # Apply our own norm
        return rmsd


def normalize_nonpbc_atoms(atoms1, atoms2):
    atoms1, atoms2 = atoms1.copy(), atoms2.copy()

    pbc1_c = atoms1.get_pbc()
    pbc2_c = atoms2.get_pbc()

    assert all(pbc1_c == pbc2_c)

    cell2_cv = atoms2.get_cell()
    cell2_cv[~pbc2_c] = atoms1.get_cell()[~pbc1_c]
    atoms2.set_cell(cell2_cv)
    return atoms1, atoms2


def are_structures_duplicates(atoms1, atoms2, symprec=1e-5, rmsd_tol=0.3,
                              adaptor=None, matcher=None):
    """Return true if atoms1 and atoms2 are duplicates."""
    import spglib
    from asr.setup.symmetrize import atomstospgcell as ats
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.alchemy.filters import RemoveExistingFilter

    if adaptor is None:
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

    if matcher is None:
        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)

    atoms1, atoms2 = normalize_nonpbc_atoms(atoms1, atoms2)

    # struc1 = adaptor.get_structure(atoms1)
    # struc2 = adaptor.get_structure(atoms2)
    # rmdup = RemoveExistingFilter([struc2],
    #                              matcher,
    #                              symprec=symprec)
    # is_duplicate = not rmdup.test(struc1)

    # Manually fix normalization
    dataset1 = spglib.get_symmetry_dataset(ats(atoms1), symprec=symprec)
    dataset2 = spglib.get_symmetry_dataset(ats(atoms2), symprec=symprec)
    if dataset1['number'] == dataset2['number']:
        rmsd = get_rmsd(atoms1, atoms2, matcher=matcher, adaptor=adaptor)
        if rmsd is None:
            is_duplicate = False
        else:
            is_duplicate = rmsd < rmsd_tol
    else:
        is_duplicate = False

    return is_duplicate


def check_duplicates(structure=None, row=None, db=None,
                     comparison_keys=[],
                     exclude_ids=set(),
                     extra_data={},
                     verbose=False,
                     symprec=1e-5,
                     rmsd_tol=0.3):
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
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)
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
                                                         symprec=symprec,
                                                         rmsd_tol=rmsd_tol)
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
