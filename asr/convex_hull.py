"""Convex hull stability analysis."""
from pathlib import Path
from collections import Counter
from typing import List, Dict

from asr.core import command, argument
from asr.paneldata import ConvexHullResult

from ase.io import read
from ase.db import connect
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow
from ase.formula import Formula

# from matplotlib.legend_handler import HandlerPatch
# from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


known_methods = ['DFT', 'DFT+D3']


@command('asr.convex_hull',
         requires=['results-asr.structureinfo.json',
                   'results-asr.database.material_fingerprint.json'],
         dependencies=['asr.structureinfo',
                       'asr.database.material_fingerprint'],
         returns=ConvexHullResult)
@argument('databases', nargs=-1, type=str)
def main(databases: List[str]) -> ConvexHullResult:
    """Calculate convex hull energies.

    It is assumed that the first database supplied is the one containing the
    standard references.

    For a database to be a valid reference database each row has to have a
    "uid" key-value-pair. Additionally, it is required that the metadata of
    each database contains following keys:

        - title: Title of the reference database.
        - legend: Collective label for all references in the database to
          put on the convex hull figures.
        - name: f-string from which to derive name for a material.
        - link: f-string from which to derive an url for a material
          (see further information below).
        - label: f-string from which to derive a material specific name to
          put on convex hull figure.
        - method: String denoting the method that was used to calculate
          reference energies. Currently accepted strings: ['DFT', 'DFT+D3'].
          "DFT" means bare DFT references energies. "DFT+D3" indicate that the
          reference also include the D3 dispersion correction.
        - energy_key (optional): Indicates the key-value-pair that represents
          the total energy of a material from. If not specified the
          default value of 'energy' will be used.

    The name and link keys are given as f-strings and can this refer to
    key-value-pairs in the given database. For example, valid metadata looks
    like:

    .. code-block:: javascript

        {
            "title": "Bulk reference phases",
            "legend": "Bulk",
            "name": "{row.formula}",
            "link": "https://cmrdb.fysik.dtu.dk/oqmd12/row/{row.uid}",
            "label": "{row.formula}",
            "method": "DFT",
            "energy_key": "total_energy"
        }

    Parameters
    ----------
    databases : list of str
        List of filenames of databases.

    """
    from asr.core import read_json

    atoms = read('structure.json')

    # TODO: Make separate recipe for calculating vdW correction to total energy
    energy = atoms.get_potential_energy()

    for filename in ['results-asr.relax.json', 'results-asr.gs.json']:
        if Path(filename).is_file():
            results = read_json(filename)
            energy_asr = results.get('etot')
            usingd3 = results.metadata.params.get('d3', False)
            if usingd3:
                print(f'Detected vdW dispersion corrections (D3) in {filename}.',
                      'Make sure you are using a DFT+D3 reference database if the',
                      'the energy in structure.json was calculated with D3.')
            if np.abs(energy_asr - energy) / len(atoms) > 0.01:
                print(f'WARNING: The energy in structure.json ({energy})',
                      f'is significantly different from the energy in {filename}',
                      f'({energy_asr}).')

    formula = atoms.get_chemical_formula()
    count = Counter(atoms.get_chemical_symbols())

    dbdata = {}
    reqkeys = {'title', 'legend', 'name', 'link', 'label'}
    for database in databases:
        # Connect to databases and save relevant rows
        refdb = connect(database)
        metadata = refdb.metadata
        assert not (reqkeys - set(metadata)), \
            'Missing some essential metadata keys.'

        rows = []
        # Select only references which contain relevant elements
        rows.extend(select_references(refdb, set(count)))
        dbdata[database] = {'rows': rows,
                            'metadata': metadata}

    ref_database = databases[0]
    ref_metadata = dbdata[ref_database]['metadata']
    ref_energy_key = ref_metadata.get('energy_key', 'energy')
    ref_energies_per_atom = get_singlespecies_reference_energies_per_atom(
        atoms, ref_database, energy_key=ref_energy_key)

    # Make a list of the relevant references
    references = []
    for data in dbdata.values():
        metadata = data['metadata']
        energy_key = metadata.get('energy_key', 'energy')
        for row in data['rows']:
            hformref = hof(row[energy_key],
                           row.count_atoms(), ref_energies_per_atom)
            reference = {'hform': hformref,
                         'formula': row.formula,
                         'uid': row.uid,
                         'natoms': row.natoms}
            reference.update(metadata)
            if 'name' in reference:
                reference['name'] = reference['name'].format(row=row)
            if 'label' in reference:
                reference['label'] = reference['label'].format(row=row)
            if 'link' in reference:
                reference['link'] = reference['link'].format(row=row)
            references.append(reference)

    assert len(atoms) == len(Formula(formula))
    return calculate_hof_and_hull(formula, energy, references,
                                  ref_energies_per_atom)


def calculate_hof_and_hull(
        formula, energy, references, ref_energies_per_atom):
    formula = Formula(formula)

    species_counts = formula.count()

    hform = hof(energy,
                species_counts,
                ref_energies_per_atom)

    pdrefs = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))

    results = {'hform': hform,
               'references': references}

    pd = PhaseDiagram(pdrefs, verbose=False)
    e0, indices, coefs = pd.decompose(str(formula))
    ehull = hform - e0 / len(formula)
    if len(species_counts) == 1:
        assert abs(ehull - hform) < 1e-10

    results['indices'] = indices.tolist()
    results['coefs'] = coefs.tolist()

    results['ehull'] = ehull
    results['thermodynamic_stability_level'] = stability_rating(hform, ehull)
    return ConvexHullResult(data=results)


LOW = 1
MEDIUM = 2
HIGH = 3
stability_names = {LOW: 'LOW', MEDIUM: 'MEDIUM', HIGH: 'HIGH'}
stability_descriptions = {
    LOW: 'Heat of formation > 0.2 eV/atom',
    MEDIUM: 'convex hull + 0.2 eV/atom < Heat of formation < 0.2 eV/atom',
    HIGH: 'Heat of formation < convex hull + 0.2 eV/atom'}


def stability_rating(hform, energy_above_hull):
    assert hform <= energy_above_hull
    if 0.2 < hform:
        return LOW
    if 0.2 < energy_above_hull:
        return MEDIUM
    return HIGH


def get_singlespecies_reference_energies_per_atom(
        atoms, references, energy_key='energy'):

    # Get reference energies
    ref_energies_per_atom = {}
    refdb = connect(references)
    for row in select_references(refdb, set(atoms.symbols)):
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            e_ref = row[energy_key] / row.natoms
            assert symbol not in ref_energies_per_atom
            ref_energies_per_atom[symbol] = e_ref

    return ref_energies_per_atom


def hof(energy, count, ref_energies_per_atom):
    """Heat of formation."""
    energy = energy - sum(n * ref_energies_per_atom[symbol]
                          for symbol, n in count.items())
    return energy / sum(count.values())


def select_references(db, symbols):
    refs: Dict[int, 'AtomsRow'] = {}

    for symbol in symbols:
        for row in db.select(symbol):
            for symb in row.count_atoms():
                if symb not in symbols:
                    break
            else:
                uid = row.get('uid')
                refs[uid] = row
    return list(refs.values())


if __name__ == '__main__':
    main.cli()
