"""Convex hull stability analysis."""
from collections import Counter
from typing import List, Dict, Any, Optional
from pathlib import Path
import functools

from asr.core import command, argument, ASRResult, prepare_result
from asr.database.browser import (
    fig, table, describe_entry, dl, br, make_panel_description
)

from ase.db import connect
from ase.io import read
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow

known_methods = ['DFT', 'DFT+D3']


panel_description = make_panel_description(
    """The heat of formation (ΔH) is the internal energy of a compound relative to
the standard states of the constituent elements at T=0 K.  The energy above the
convex hull is the internal energy relative to the most stable (possibly mixed)
phase of the constituent elements at T=0 K.""",
    articles=['C2DB'],
)


def webpanel(result, row, key_descriptions):
    hulltable1 = table(row,
                       'Stability',
                       ['hform', 'ehull'],
                       key_descriptions)
    hulltables = convex_hull_tables(row)
    panel = {'title': describe_entry('Thermodynamic stability', panel_description),
             'columns': [[fig('convex-hull.png')],
                         [hulltable1] + hulltables],
             'plot_descriptions': [{'function':
                                    functools.partial(plot, thisrow=row),
                                    'filenames': ['convex-hull.png']}],
             'sort': 1}

    thermostab = row.get('thermodynamic_stability_level')
    stabilities = {1: 'low', 2: 'medium', 3: 'high'}
    high = 'Heat of formation < convex hull + 0.2 eV/atom'
    medium = 'Heat of formation < 0.2 eV/atom'
    low = 'Heat of formation > 0.2 eV/atom'
    thermodynamic = describe_entry(
        'Thermodynamic',
        'Classifier for the thermodynamic stability of a material.'
        + br
        + dl(
            [
                ['LOW', low],
                ['MEDIUM', medium],
                ['HIGH', high],
            ]
        )
    )
    row = [thermodynamic,
           stabilities[thermostab].upper()]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', ''],
                             'rows': [row],
                             'columnwidth': 3}]],
               'sort': 1}
    return [panel, summary]


# class Reference(TypedDict):
#     """Container for information on a reference."""

#     hform: float
#     formula: str
#     uid: str
#     natoms: int
#     name: str
#     label: str
#     link: str


@prepare_result
class Result(ASRResult):

    ehull: float
    hform: float
    references: List[dict]
    thermodynamic_stability_level: str
    coefs: Optional[List[float]]
    indices: Optional[List[int]]
    key_descriptions = {
        "ehull": "Energy above convex hull [eV/atom].",
        "hform": "Heat of formation [eV/atom].",
        "thermodynamic_stability_level": "Thermodynamic stability level.",
        "references": "List of relevant references.",
        "indices":
        "Indices of references that this structure will decompose into.",
        "coefs": "Fraction of decomposing references (see indices doc).",
    }

    formats = {"ase_webpanel": webpanel}


@command('asr.convex_hull',
         requires=['results-asr.structureinfo.json',
                   'results-asr.database.material_fingerprint.json'],
         dependencies=['asr.structureinfo',
                       'asr.database.material_fingerprint'],
         returns=Result)
@argument('databases', nargs=-1, type=str)
def main(databases: List[str]) -> Result:
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
            'title': 'Bulk reference phases',
            'legend': 'Bulk',
            'name': '{row.formula}',
            'link': 'https://cmrdb.fysik.dtu.dk/oqmd12/row/{row.uid}',
            'label': '{row.formula}',
            'method': 'DFT',
            'energy_key': 'total_energy',
        }

    Parameters
    ----------
    databases : list of str
        List of filenames of databases.

    """
    from asr.relax import main as relax
    from asr.gs import main as groundstate
    from asr.core import read_json
    atoms = read('structure.json')

    if not relax.done:
        if not groundstate.done:
            groundstate()

    # TODO: Make separate recipe for calculating vdW correction to total energy
    for filename in ['results-asr.relax.json', 'results-asr.gs.json']:
        if Path(filename).is_file():
            results = read_json(filename)
            energy = results.get('etot')
            usingd3 = results.metadata.params.get('d3', False)
            break

    if usingd3:
        mymethod = 'DFT+D3'
    else:
        mymethod = 'DFT'

    formula = atoms.get_chemical_formula()
    count = Counter(atoms.get_chemical_symbols())

    dbdata = {}
    reqkeys = {'title', 'legend', 'name', 'link', 'label', 'method'}
    for database in databases:
        # Connect to databases and save relevant rows
        refdb = connect(database)
        metadata = refdb.metadata
        assert not (reqkeys - set(metadata)), \
            'Missing some essential metadata keys.'

        dbmethod = metadata['method']
        assert dbmethod in known_methods, f'Unknown method: {dbmethod}'
        assert dbmethod == mymethod, \
            ('You are using a reference database with '
             f'inconsistent methods: {mymethod} (this material) != '
             f'{dbmethod} ({database})')

        rows = []
        # Select only references which contain relevant elements
        rows.extend(select_references(refdb, set(count)))
        dbdata[database] = {'rows': rows,
                            'metadata': metadata}

    ref_database = databases[0]
    ref_metadata = dbdata[ref_database]['metadata']
    ref_energy_key = ref_metadata.get('energy_key', 'energy')
    ref_energies = get_reference_energies(atoms, ref_database,
                                          energy_key=ref_energy_key)
    hform = hof(energy,
                count,
                ref_energies)
    # Make a list of the relevant references
    references = []
    for data in dbdata.values():
        metadata = data['metadata']
        energy_key = metadata.get('energy_key', 'energy')
        for row in data['rows']:
            hformref = hof(row[energy_key],
                           row.count_atoms(), ref_energies)
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

    pdrefs = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))

    results = {'hform': hform,
               'references': references}

    if len(count) == 1:
        ehull = hform
        results['indices'] = None
        results['coefs'] = None
    else:
        pd = PhaseDiagram(pdrefs, verbose=False)
        e0, indices, coefs = pd.decompose(formula)
        ehull = hform - e0 / len(atoms)
        results['indices'] = indices.tolist()
        results['coefs'] = coefs.tolist()

    results['ehull'] = ehull

    if hform >= 0.2:
        thermodynamic_stability = 1
    elif hform is None or ehull is None:
        thermodynamic_stability = None
    elif ehull >= 0.2:
        thermodynamic_stability = 2
    else:
        thermodynamic_stability = 3

    results['thermodynamic_stability_level'] = thermodynamic_stability
    return Result(data=results)


def get_reference_energies(atoms, references, energy_key='energy'):
    count = Counter(atoms.get_chemical_symbols())

    # Get reference energies
    ref_energies = {}
    refdb = connect(references)
    for row in select_references(refdb, set(count)):
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            e_ref = row[energy_key] / row.natoms
            assert symbol not in ref_energies
            ref_energies[symbol] = e_ref

    return ref_energies


def hof(energy, count, ref_energies):
    """Heat of formation."""
    energy = energy - sum(n * ref_energies[symbol]
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


def plot(row, fname, thisrow):
    from ase.phasediagram import PhaseDiagram
    import matplotlib.pyplot as plt

    data = row.data['results-asr.convex_hull.json']

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    references = data['references']
    pdrefs = []
    legends = []
    colors = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))
        if reference['legend'] not in legends:
            legends.append(reference['legend'])
        idlegend = legends.index(reference['legend'])
        colors.append(f'C{idlegend + 2}')

    pd = PhaseDiagram(pdrefs, verbose=False)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()

    for it, legend in enumerate(legends):
        ax.scatter([], [], facecolor='none', marker='o',
                   edgecolor=f'C{it + 2}', label=legend)

    if len(count) == 2:
        x, e, _, hull, simplices, xlabel, ylabel = pd.plot2d2()
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-', color='C0')
        names = [ref['label'] for ref in references]
        if row.hform < 0:
            mask = e < 0.05
            e = e[mask]
            x = x[mask]
            hull = hull[mask]
            names = [name for name, m in zip(names, mask) if m]
        ax.scatter(x, e, facecolor='none', marker='o', edgecolor=colors)

        delta = e.ptp() / 30
        for a, b, name, on_hull in zip(x, e, names, hull):
            va = 'center'
            ha = 'left'
            dy = 0
            dx = 0.02
            ax.text(a + dx, b + dy, name, ha=ha, va=va)

        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')

        # Circle this material
        xt = count.get(B, 0) / sum(count.values())
        ax.plot([xt], [row.hform], 'o', color='C1', label=f'{thisrow.formula}')
        ymin = e.min()

        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * delta)
        plt.legend(loc='lower left')
    else:
        x, y, _, hull, simplices = pd.plot2d3()
        names = [ref['label'] for ref in references]
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        ax.scatter(x, y, facecolor='none', marker='o', edgecolor=colors)

        for a, b, name, on_hull in zip(x, y, names, hull):
            if on_hull:
                ax.text(a - 0.02, b, name, ha='right', va='top')
        A, B, C = pd.symbols
        bfrac = count.get(B, 0) / sum(count.values())
        cfrac = count.get(C, 0) / sum(count.values())

        ax.plot([bfrac + cfrac / 2],
                [cfrac * 3**0.5 / 2],
                'o', color='C1', label=f'{thisrow.formula}')
        plt.legend(loc='upper left')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def convex_hull_tables(row: AtomsRow) -> List[Dict[str, Any]]:
    data = row.data['results-asr.convex_hull.json']

    references = data.get('references', [])
    tables = {}
    for reference in references:
        tables[reference['title']] = []

    for reference in sorted(references, reverse=True,
                            key=lambda x: x['hform']):
        name = reference['name']
        matlink = reference['link']
        if reference['uid'] != row.uid:
            name = f'<a href="{matlink}">{name}</a>'
        e = reference['hform']
        tables[reference['title']].append([name, '{:.2f} eV/atom'.format(e)])

    final_tables = []
    for title, rows in tables.items():
        final_tables.append({'type': 'table',
                             'header': [title, ''],
                             'rows': rows})
    return final_tables


if __name__ == '__main__':
    main.cli()

