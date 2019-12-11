from asr.core import command, option
from asr.borncharges import get_polarization_phase, get_wavefunctions

from .gs import calculate


@command()
@option('--gpwname', help='Formal polarization gpw file name')
@option('--kptdensity', help='Kpoint density for gpw file')
def main(gpwname='formalpol.gpw', kptdensity=12.0, gsresults=calculate):
    from pathlib import Path
    from gpaw import GPAW
    from gpaw.mpi import world
    calc = GPAW('gs.gpw', txt=None)
    params = calc.parameters
    atoms = calc.atoms
    calc = get_wavefunctions(atoms=atoms, name=gpwname,
                             params=params, density=kptdensity)
    phase_c = get_polarization_phase(calc)

    results = {'phase_c': phase_c,
               '__key_descriptions__':
               {'phase_c': 'Formal polarization phase'}}

    world.barrier()
    if world.rank == 0:
        f = Path(gpwname)
        if f.is_file():
            f.unlink()

    return results


if __name__ == '__main__':
    main.cli()
