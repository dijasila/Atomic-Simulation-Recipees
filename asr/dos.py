"""Density of states."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from asr.core import command, option
from asr.paneldata import DOSResult

@command('asr.dos',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'])
@option('--name', type=str)
@option('--kptdensity', help='K-point density', type=float)
def main(name: str = 'dos.gpw',
         kptdensity: float = 12.0) -> DOSResult:
    """Calculate DOS."""
    from gpaw import GPAW

    path = Path(name)
    if not path.is_file():
        calc = GPAW(path.with_name('gs.gpw'),
                    txt=path.with_name('dos.txt')).fixed_density(
            kpts={'density': kptdensity},
            nbands='300%',
            convergence={'bands': -10})
        calc.write(path)

    calc = GPAW(path)
    doscalc = calc.dos()
    data = _main(doscalc)
    data['natoms'] = len(calc.atoms)
    data['volume'] = calc.atoms.get_volume()
    return DOSResult(data=data)


def _main(doscalc) -> dict:
    energies_e = np.linspace(-10, 10, 201)
    data = {'energies_e': energies_e.tolist(),
            'dosspin1_e': []}
    for spin in range(doscalc.nspins):
        dos_e = doscalc.raw_dos(energies_e, spin, width=0)
        data[f'dosspin{spin}_e'] = dos_e.tolist()
    return data
