"""Magnetic anisotropy."""
from math import pi
from asr.core import command, read_json
from asr.result.resultdata import MagAniResult


def get_spin_axis():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    return anis['theta'] * 180 / pi, anis['phi'] * 180 / pi


def get_spin_index():
    anis = read_json('results-asr.magnetic_anisotropy.json')
    axis = anis['spin_axis']
    if axis == 'z':
        index = 2
    elif axis == 'y':
        index = 1
    else:
        index = 0
    return index


def spin_axis(theta, phi):
    import numpy as np
    if theta == 0:
        return 'z'
    elif np.allclose(phi, 90):
        return 'y'
    else:
        return 'x'


params = '''asr.gs@calculate:calculator +{'mode':'lcao','kpts':(2,2,2)}'''
tests = [{'cli': ['ase build -x hcp Co structure.json',
                  f'asr run "setup.params {params}"',
                  'asr run asr.magnetic_anisotropy',
                  'asr run database.fromtree',
                  'asr run "database.browser --only-figures"']}]


@command('asr.magnetic_anisotropy',
         tests=tests,
         returns=MagAniResult,
         dependencies=['asr.gs@calculate', 'asr.magstate'])
def main() -> MagAniResult:
    """Calculate the magnetic anisotropy.

    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Returns
    -------
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
    """
    from asr.core import read_json
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from gpaw import GPAW

    magstateresults = read_json('results-asr.magstate.json')
    magstate = magstateresults['magstate']

    # Figure out if material is magnetic
    results = {}

    if magstate == 'NM':
        results['E_x'] = 0
        results['E_y'] = 0
        results['E_z'] = 0
        results['dE_zx'] = 0
        results['dE_zy'] = 0
        results['theta'] = 0
        results['phi'] = 0
        results['spin_axis'] = 'z'
        return MagAniResult(data=results)

    calc = GPAW('gs.gpw')
    width = 0.001
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})
    Ex, Ey, Ez = (soc_eigenstates(calc,
                                  theta=theta, phi=phi,
                                  occcalc=occcalc).calculate_band_energy()
                  for theta, phi in [(90, 0), (90, 90), (0, 0)])

    dE_zx = Ez - Ex
    dE_zy = Ez - Ey

    DE = max(dE_zx, dE_zy)
    theta = 0
    phi = 0
    if DE > 0:
        theta = 90
        if dE_zy > dE_zx:
            phi = 90

    axis = spin_axis(theta, phi)

    results.update({'spin_axis': axis,
                    'theta': theta / 180 * pi,
                    'phi': phi / 180 * pi,
                    'E_x': Ex * 1e3,
                    'E_y': Ey * 1e3,
                    'E_z': Ez * 1e3,
                    'dE_zx': dE_zx * 1e3,
                    'dE_zy': dE_zy * 1e3})
    return MagAniResult(data=results)


if __name__ == '__main__':
    main.cli()
