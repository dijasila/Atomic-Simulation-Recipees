from asr.core import command, option


@command(module='asr.analyze_state',
         requires=['gs.gpw'],
         resources='1:1h')
def main():
    """Write out wavefunction and analyze it.

    This recipe reads in an existing gs.gpw file and writes out wavefunctions
    of different states (either the one of a specific given bandindex or of
    all the defect states in the gap). Furthermore, it will feature some post
    analysis on those states.
    """
    from gpaw import restart

    atoms, calc = restart('gs.gpw', txt=None)

    return None
