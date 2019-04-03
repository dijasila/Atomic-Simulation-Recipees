import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any

import click

from ase.db import connect
from ase.io import read
from ase.phasediagram import PhaseDiagram
from ase.db.row import AtomsRow


@click.command()
@click.option('-r', '--references', type=str,
              help='Reference database.',
              default='references.db')
def main(references: str):
    db = connect(references)
    atoms = read('gs.gpw')
    count = Counter(atoms.get_chemical_symbols())
    refs: List[Tuple[str, float]] = []
    for symbol in count:
        for row in db.select(symbol):
            refs.append((row.formula, row.energy / row.natoms))
    convex_hull(atoms, refs)


def convex_hull(atoms, references):
    formula = atoms.get_chemical_formula()
    energy = atoms.get_potential_energy()

    try:
        pd = PhaseDiagram(references, filter=formula)
    except ValueError:
        return

    N = len(atoms)
    e0, _, _ = pd.decompose(formula)
    ehull = (energy - e0) / N

    refs2 = []
    for i, (count, e, name, natoms) in enumerate(pd.references):
        refs2.append((name, pd.points[i, -1] * natoms))

    Path('convex_hull.json').write_text(
        json.dumps({'ehull': ehull,
                    'references': refs2}))


def collect_data(atoms):
    if not Path('prototype.json').is_file():
        return {}, {}, {}
    data = json.loads(Path('prototype.json'))
    return ({'ehull', data.pop('ehull')},
            [('ehull', '?', '', 'eV/atom')],
            data)


def plot(row, fname):
    from ase.phasediagram import PhaseDiagram, parse_formula
    import matplotlib.pyplot as plt

    data = row.data.get('chdata')
    if data is None or row.data.get('references') is None:
        return

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    refs = data['refs']
    pd = PhaseDiagram(refs, verbose=False)

    fig = plt.figure()
    ax = fig.gca()

    if len(count) == 2:
        x, e, names, hull, simplices, xlabel, ylabel = pd.plot2d2()
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-', color='lightblue')
        ax.plot(x, e, 's', color='C0', label='Bulk')
        dy = e.ptp() / 30
        for a, b, name in zip(x, e, names):
            ax.text(a, b - dy, name, ha='center', va='top')
        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')
        label = '2D'
        ymin = e.min()
        for y, formula, prot, magstate, id, uid in row.data.references:
            count = parse_formula(formula)[0]
            x = count[B] / sum(count.values())
            if id == row.id:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
            ax.text(x + 0.03, y, '{}-{}'.format(prot, magstate))
            ymin = min(ymin, y)
        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * dy)
    else:
        x, y, names, hull, simplices = pd.plot2d3()
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        ax.plot(x[hull], y[hull], 's', color='C0', label='Bulk (on hull)')
        ax.plot(x[~hull], y[~hull], 's', color='C2', label='Bulk (above hull)')
        for a, b, name in zip(x, y, names):
            ax.text(a - 0.02, b, name, ha='right', va='top')
        A, B, C = pd.symbols
        label = '2D'
        for e, formula, prot, magstate, id, uid in row.data.references:
            count = parse_formula(formula)[0]
            x = count.get(B, 0) / sum(count.values())
            y = count.get(C, 0) / sum(count.values())
            x += y / 2
            y *= 3**0.5 / 2
            if id == row.id:
                ax.plot([x], [y], 'rv', label=label)
                ax.plot([x], [y], 'ko', ms=15, fillstyle='none')
            else:
                ax.plot([x], [y], 'v', color='C1', label=label)
            label = None
        plt.axis('off')

    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def convex_hull_tables(row: AtomsRow,
                       project: str = 'c2db',
                       ) -> 'Tuple[Dict[str, Any], Dict[str, Any]]':
    if row.data.get('references') is None:
        return None, None
    from ase.symbols import string2symbols

    rows = []
    for e, formula, prot, magstate, id, uid in sorted(row.data.references,
                                                      reverse=True):
        name = '{} ({}-{})'.format(formula, prot, magstate)
        if id != row.id:
            name = '<a href="/{}/row/{}">{}</a>'.format(project, uid, name)
        rows.append([name, '{:.3f} eV/atom'.format(e)])

    refs = row.data.get('chdata')['refs']
    bulkrows = []
    for formula, e in refs:
        e /= len(string2symbols(formula))
        link = '<a href="/oqmd12/row/{formula}">{formula}</a>'.format(
            formula=formula)
        bulkrows.append([link, '{:.3f} eV/atom'.format(e)])

    return ({'type': 'table',
             'header': ['Monolayer formation energies', ''],
             'rows': rows},
            {'type': 'table',
             'header': ['Bulk formation energies', ''],
             'rows': bulkrows})


def webpanel(row, key_descriptions):
    from asr.custom import fig
    from asr.custom import table

    prefix = key_descriptions.get('prefix', '')
    if 'c2db-' in prefix:  # make sure links to other rows just works!
        projectname = 'c2db'
    else:
        projectname = 'default'

    hulltable1 = table('Property',
                       ['hform', 'ehull', 'minhessianeig'],
                       key_descriptions)
    hulltable2, hulltable3 = convex_hull_tables(row, projectname)

    panel = ('Stability',
             [[fig('convex-hull.png')],
              [hulltable1, hulltable2, hulltable3]])

    things = [(convex_hull, ['convex-hull.png'])]

    return panel, things


group = 'Property'
dependencies = ['asr.gs']


if __name__ == '__main__':
    main()
