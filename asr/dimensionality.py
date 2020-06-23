from asr.core import command, argument, AtomsFile
from ase import Atoms


@command('asr.dimensionality')
@argument('atoms', type=AtomsFile())
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
    return {'k_intervals': k_intervals}
