"""Effective Born charges."""
import numpy as np
from asr.core import command, option
from asr.paneldata import BornChargesResult

@command('asr.borncharges',
         dependencies=['asr.gs@calculate'],
         requires=['gs.gpw'],
         returns=BornChargesResult)
@option('--displacement', help='Atomic displacement (Å)', type=float)
def main(displacement: float = 0.01) -> BornChargesResult:
    """Calculate Born charges."""
    from gpaw import GPAW

    from ase.units import Bohr

    from asr.core import chdir, read_json
    from asr.formalpolarization import main as formalpolarization
    from asr.setup.displacements import main as setupdisplacements
    from asr.setup.displacements import get_all_displacements, get_displacement_folder

    if not setupdisplacements.done:
        setupdisplacements(displacement=displacement)

    calc = GPAW('gs.gpw', txt=None)
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    Z_avv = []
    phase_ascv = np.zeros((len(atoms), 2, 3, 3), float)

    for ia, iv, sign in get_all_displacements(atoms):
        folder = get_displacement_folder(ia, iv, sign, displacement)

        with chdir(folder):
            if not formalpolarization.done:
                formalpolarization()

        polresults = read_json(folder / 'results-asr.formalpolarization.json')
        phase_c = polresults['phase_c']
        isign = [None, 1, 0][sign]
        phase_ascv[ia, isign, :, iv] = phase_c

    for phase_scv in phase_ascv:
        dphase_cv = (phase_scv[1] - phase_scv[0])
        mod_cv = np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dphase_cv -= mod_cv
        phase_scv[1] -= mod_cv
        dP_vv = (np.dot(dphase_cv.T, cell_cv).T
                 / (2 * np.pi * vol))
        Z_vv = dP_vv * vol / (2 * displacement / Bohr)
        Z_avv.append(Z_vv)

    Z_avv = np.array(Z_avv)
    data = {'Z_avv': Z_avv, 'sym_a': sym_a}

    return data


if __name__ == '__main__':
    main.cli()
