"""Piezoelectric tensor.

Module containing functionality for calculating the piezoelectric
tensor. The central recipe of this module is
:func:`asr.piezoelectrictensor.main`.

"""
from asr.core import command, option, DictStr
from asr.paneldata import PiezoEleTenResult


@command(module="asr.piezoelectrictensor",
         returns=PiezoEleTenResult)
@option('--strain-percent', help='Strain fraction.', type=float)
@option('--calculator', help='Calculator parameters.', type=DictStr())
def main(strain_percent: float = 1,
         calculator: dict = {
             'name': 'gpaw',
             'mode': {'name': 'pw', 'ecut': 800},
             'xc': 'PBE',
             'basis': 'dzp',
             'kpts': {'density': 12.0},
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'convergence': {'eigenstates': 1e-11,
                             'density': 1e-7},
             'symmetry': 'off',
             'txt': 'formalpol.txt',
             'charge': 0
         }) -> PiezoEleTenResult:
    """Calculate piezoelectric tensor.

    This recipe calculates the clamped and full piezoelectric
    tensor. You generally will only need the full piezoelectric
    tensor. The clamped piezoelectric tensor is useful for analyzing
    results. The piezoelectric tensor is calculated using a finite
    difference scheme by calculating the derivative of the
    polarization density at finite strains.

    Parameters
    ----------
    strain_percent : float
        Amount of strain applied to the material.
    calculator : dict
        Calculator parameters.

    """
    import numpy as np
    from ase.calculators.calculator import kptdensity2monkhorstpack
    from ase.io import read
    from ase.units import Bohr
    from asr.core import read_json, chdir
    from asr.formalpolarization import main as formalpolarization
    from asr.relax import main as relax
    from asr.setup.strains import main as setupstrains
    from asr.setup.strains import clamped as setupclampedstrains
    from asr.setup.strains import get_relevant_strains, get_strained_folder_name

    if not setupstrains.done:
        setupstrains(strain_percent=strain_percent)

    if not setupclampedstrains.done:
        setupclampedstrains(strain_percent=strain_percent)

    atoms = read("structure.json")

    # From experience it is important to use
    # non-gamma centered grid when using symmetries.
    # Might have something to do with degeneracies, not sure.
    if 'density' in calculator['kpts']:
        kpts = calculator['kpts']
        density = kpts.pop('density')
        kpts['size'] = kptdensity2monkhorstpack(atoms, density, True)

    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    pbc_c = atoms.get_pbc()
    if not all(pbc_c):
        N = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    else:
        N = 1.0
    eps_clamped_vvv = np.zeros((3, 3, 3), float)
    eps_vvv = np.zeros((3, 3, 3), float)
    ij = get_relevant_strains(atoms.pbc)

    for clamped in [True, False]:
        for i, j in ij:
            phase_sc = np.zeros((2, 3), float)
            for s, sign in enumerate([-1, 1]):
                folder = get_strained_folder_name(sign * strain_percent, i, j,
                                                  clamped=clamped)
                with chdir(folder):
                    if not clamped and not relax.done:
                        relax.cli([])
                    if not formalpolarization.done:
                        formalpolarization(calculator=calculator)

                polresults = read_json(folder / 'results-asr.formalpolarization.json')
                phase_sc[s] = polresults['phase_c']

            dphase_c = phase_sc[1] - phase_sc[0]
            dphase_c -= np.round(dphase_c / (2 * np.pi)) * 2 * np.pi
            dphasedeps_c = dphase_c / (2 * strain_percent * 0.01)
            eps_v = (np.dot(dphasedeps_c, cell_cv)
                     / (2 * np.pi * vol))
            eps_v *= N / Bohr

            if clamped:
                epsref_vvv = eps_clamped_vvv
            else:
                epsref_vvv = eps_vvv

            epsref_vvv[:, i, j] = eps_v
            epsref_vvv[:, j, i] = eps_v

    data = {'eps_vvv': eps_vvv,
            'eps_clamped_vvv': eps_clamped_vvv}

    return data


if __name__ == '__main__':
    main.cli()
