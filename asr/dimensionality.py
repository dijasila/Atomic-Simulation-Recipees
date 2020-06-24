from asr.core import command, option, AtomsFile
from ase import Atoms


@command('asr.dimensionality')
@option('--atoms', type=AtomsFile(), default='structure.json')
def main(atoms: Atoms):
    """Make cluster and dimensionality analysis of the input structure.

    Analyzes the primary dimensionality of the input structure
    and analyze clusters following Mahler, et. al.
    Physical Review Materials 3 (3), 034003.
    """
    from ase.geometry.dimensionality import analyze_dimensionality
    k_intervals = [dict(interval._asdict())
                   for interval in
                   analyze_dimensionality(atoms)]

    # Fix for numpy.int64 in cdim which is not jsonable.
    for interval in k_intervals:
        cdim = {int(key): value for key, value in interval['cdim'].items()}
        interval['cdim'] = cdim

    results = {'k_intervals': k_intervals}
    primary_interval = k_intervals[0]
    dim_primary = primary_interval['dimtype']
    dim_primary_score = primary_interval['score']

    results['dim_primary'] = dim_primary
    results['dim_primary_score'] = dim_primary_score
    for nd in range(4):
        results[f'dim_nclusters_{nd}D'] = primary_interval['h'][nd]

    return results
