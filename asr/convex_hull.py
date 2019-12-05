from collections import Counter
from typing import List, Dict, Any

from asr.core import command, argument, option

from ase.db import connect
from ase.io import read
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow


def webpanel(row, key_descriptions):
    from asr.browser import fig, table

    hulltable1 = table(row,
                       'Stability',
                       ['hform', 'ehull'],
                       key_descriptions)
    hulltables = convex_hull_tables(row)
    panel = {'title': 'Thermodynamic stability',
             'columns': [[fig('convex-hull.png')],
                         [hulltable1] + hulltables],
             'plot_descriptions': [{'function': plot,
                                    'filenames': ['convex-hull.png']}],
             'sort': 1}

    thermostab = row.get('thermodynamic_stability_level')
    stabilities = {1: 'low', 2: 'medium', 3: 'high'}
    high = 'Heat of formation < convex hull + 0.2 eV/atom'
    medium = 'Heat of formation < 0.2 eV/atom'
    low = 'Heat of formation > 0.2 eV/atom'
    row = ['Thermal',
           '<a href="#" data-toggle="tooltip" data-html="true" ' +
           'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
               low, medium, high, stabilities[thermostab].upper())]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', ''],
                             'rows': [row]}]]}
    return [panel, summary]


@command('asr.convex_hull',
         requires=['results-asr.gs.json', 'results-asr.structureinfo.json',
                   'results-asr.database.material_fingerprint.json'],
         dependencies=['asr.gs',
                       'asr.structureinfo',
                       'asr.database.material_fingerprint'],
         webpanel=webpanel)
@argument('databases', nargs=-1)
@option('--standardreferences',
        help='Database containing standard references.')
def main(databases, standardreferences=None):
    """Calculate convex hull energies

    The reference database has to have a type column indicating"""
    from asr.core import read_json
    if standardreferences is None:
        standardreferences = databases[0]

    atoms = read('structure.json')
    formula = atoms.get_chemical_formula()
    count = Counter(atoms.get_chemical_symbols())
    ref_energies = get_reference_energies(atoms, databases[0])
    hform = hof(read_json('results-asr.gs.json').get('etot'),
                count,
                ref_energies)
    mf = read_json('results-asr.database.material_fingerprint.json')
    uid = mf['uid']

    # Now compute convex hull
    dbdata = {}
    for database in databases:
        # Connect to databases and save relevant rows
        rows = []
        refdb = connect(database)
        rows.extend(select_references(refdb, set(count)))
        dbdata[database] = {'rows': rows,
                            'metadata': refdb.metadata}

    # Make a list of the relevant references
    references = []
    for data in dbdata.values():
        metadata = data['metadata']
        for row in data['rows']:
            if row.uid == uid:
                continue
            # Take the energy from the gs recipe if its calculated
            # or fall back to row.energy
            energy = row.data.get('results-asr.gs.json')['etot'] if \
                'results-asr.gs.json' in row.data else row.energy
            hformref = hof(energy, row.count_atoms(), ref_energies)
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

    pd = PhaseDiagram(pdrefs)
    e0, indices, coefs = pd.decompose(formula)

    results = {'hform': hform,
               'references': references}
    ehull = hform - e0 / len(atoms)
    results['ehull'] = ehull
    results['indices'] = indices.tolist()
    results['coefs'] = coefs.tolist()
    results['__key_descriptions__'] = {
        'ehull': 'KVP: Energy above convex hull [eV/atom]',
        'hform': 'KVP: Heat of formation [eV/atom]',
        'thermodynamic_stability_level': 'KVP: Thermodynamic stability level'}

    if hform >= 0.2:
        thermodynamic_stability = 1
    elif hform is None or ehull is None:
        thermodynamic_stability = None
    elif ehull >= 0.2:
        thermodynamic_stability = 2
    else:
        thermodynamic_stability = 3

    results['thermodynamic_stability_level'] = thermodynamic_stability
    return results


def get_reference_energies(atoms, references):
    count = Counter(atoms.get_chemical_symbols())

    # Get reference energies
    ref_energies = {}
    refdb = connect(references)
    for row in select_references(refdb, set(count)):
        if len(row.count_atoms()) == 1:
            symbol = row.symbols[0]
            e_ref = row.energy / row.natoms
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


def plot(row, fname):
    from ase.phasediagram import PhaseDiagram
    import re
    import matplotlib.pyplot as plt

    data = row.data['results-asr.convex_hull.json']

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    references = data['references']
    pdrefs = []
    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))

    pd = PhaseDiagram(pdrefs, verbose=False)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()

    if len(count) == 2:
        x, e, _, hull, simplices, xlabel, ylabel = pd.plot2d2()
        names = [re.sub(r'(\d+)', r'$_{\1}$', ref['label'])
                 for ref in references]
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-b')
        ax.plot(x, e, 'sg')
        delta = e.ptp() / 30
        for a, b, name, on_hull in zip(x, e, names, hull):
            if on_hull:
                va = 'top'
                ha = 'center'
                dy = - delta
                dx = 0
            else:
                va = 'center'
                ha = 'left'
                dy = 0
                dx = 0.02
            ax.text(a + dx, b + dy, name, ha=ha, va=va)

        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-', color='lightblue')

        # Circle this material
        xt = count.get(B, 0) / sum(count.values())
        ax.plot([xt], [row.hform], 'sk')
        ymin = e.min()

        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * delta)
    else:
        x, y, names, hull, simplices = pd.plot2d3()
        names = [re.sub(r'(\d+)', r'$_{\1}$', ref['label'])
                 for ref in references]
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        ax.plot(x[hull], y[hull], 's', color='C0', label='On hull')
        ax.plot(x[~hull], y[~hull], 's', color='C2', label='Above hull')
        for a, b, name in zip(x, y, names):
            ax.text(a - 0.02, b, name, ha='right', va='top')
        A, B, C = pd.symbols
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
        tables[reference['title']].append([name, '{:.3f} eV/atom'.format(e)])

    final_tables = []
    for title, rows in tables.items():
        final_tables.append({'type': 'table',
                             'header': [title, ''],
                             'rows': rows})
    return final_tables


if __name__ == '__main__':
    main.cli()
