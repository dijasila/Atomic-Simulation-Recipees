from asr.core import command


@command('asr.dimensionality')
def main(atoms):
    """Make cluster and dimensionality analysis of the input structure.

    Analyzes the primary dimensionality of the input structure
    and analyze clusters following Mahler, et. al.
    Physical Review Materials 3 (3), 034003.
    """
    from ase.geometry.dimensionality import analyze_dimensionality
    k_intervals = [dict(interval._asdict())
                   for interval in
                   analyze_dimensionality(atoms)]

    return {'k_intervals': k_intervals}
