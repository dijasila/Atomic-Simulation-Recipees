from asr.core import command, option, AtomsFile, ASRResult
from pathlib import Path
from ase import Atoms
from ase.db.row import AtomsRow
import numpy as np


def get_dimtypes():
    """Create a list of all dimensionality types."""
    from itertools import product
    s = set(product([0, 1], repeat=4))
    s2 = sorted(s, key=lambda x: (sum(x), *[-t for t in x]))[1:]
    string = "0123"
    return ["".join(x for x, y in zip(string, s3) if y) + "D" for s3 in s2]


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    dimtable = table(row, 'Dimensionality scores',
                     [f'dim_score_{dimtype}' for dimtype in get_dimtypes()],
                     key_descriptions, 2)
    panel = {'title': 'Dimensionality analysis',
             'columns': [[dimtable], [fig('dimensionality-histogram.png')]]}
    return [panel]


def plot_dimensionality_histogram(row: AtomsRow, path: Path) -> None:
    from matplotlib import pyplot as plt
    dimtypes = get_dimtypes()
    vs = []
    for dimtype in dimtypes:
        v = row.get(f'dim_score_{dimtype}', 0)
        vs.append(v)
    x = np.arange(dimtypes)
    fig, ax = plt.subplots()
    ax.bar(x, vs)
    ax.set_xticks(x)
    prettykeys = [f'$S_{{{dimtype}}}$' for dimtype in dimtypes]
    ax.set_xticklabels(prettykeys, rotation=90)
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()


@command('asr.dimensionality')
@option('--atoms', type=AtomsFile(), default='structure.json')
def main(atoms: Atoms) -> ASRResult:
    """Make cluster and dimensionality analysis of the input structure.

    Analyzes the primary dimensionality of the input structure and analyze
    clusters following Mahler, et. al.  Physical Review Materials 3 (3),
    034003.

    """
    from ase.geometry.dimensionality import analyze_dimensionality
    k_intervals = [dict(interval._asdict())
                   for interval in
                   analyze_dimensionality(atoms, merge=False)]

    dim_scores = {}
    dim_thresholds = {i: None for i in range(4)}  # 1000 is arbitrary
    # Fix for numpy.int64 in cdim which is not jsonable.
    for interval in k_intervals:
        cdim = {int(key): value for key, value in interval['cdim'].items()}
        interval['cdim'] = cdim
        dim_scores[interval['dimtype']] = interval['score']
        for nd in range(4):
            if interval['h'][nd]:
                dim_thresholds[nd] = min(dim_thresholds[nd] or 1000, interval['a'])

    results = {'k_intervals': k_intervals}
    primary_interval = k_intervals[0]
    dim_primary = primary_interval['dimtype']
    dim_primary_score = primary_interval['score']

    results['dim_primary'] = dim_primary
    results['dim_primary_score'] = dim_primary_score

    for dimtype in get_dimtypes():
        results[f'dim_score_{dimtype}'] = dim_scores.get(dimtype, 0)
    for nd in range(4):
        results[f'dim_nclusters_{nd}D'] = primary_interval['h'][nd]
        if dim_thresholds[nd] is not None:
            results[f'dim_threshold_{nd}D'] = dim_thresholds[nd]

    return results
