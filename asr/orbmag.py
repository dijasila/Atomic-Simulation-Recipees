"""
Module for calculating local orbital magnetic moment of each atom along the
crystal's easy axis.
"""
import numpy as np
from asr.core import command, read_json
from asr.result.resultdata import OrbMagResult


@command('asr.orbmag',
         requires=['gs.gpw'],
         returns=OrbMagResult,
         dependencies=['asr.gs@calculate',
                       'asr.magstate',
                       'asr.magnetic_anisotropy'])
def main() -> OrbMagResult:
    """Calculate local orbital magnetic moments."""
    from gpaw.new.ase_interface import GPAW
    from gpaw.spinorbit import soc_eigenstates

    magstate = read_json('results-asr.magstate.json')['magstate']

    # Figure out if material is magnetic
    if magstate == 'NM':
        results = {'orbmag_a': None,
                   'orbmag_sum': None,
                   'orbmag_max': None}
        return OrbMagResult(data=results)

    # Compute spin-orbit eigenstates non-self-consistently
    calc = GPAW('gs.gpw', txt=None)

    theta = read_json('results-asr.magnetic_anisotropy.json')['theta']
    phi = read_json('results-asr.magnetic_anisotropy.json')['phi']
    soc_eigs = soc_eigenstates(calc, theta=theta, phi=phi)

    easy_axis = np.array([np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                          np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                          np.cos(np.deg2rad(theta))])

    orbmag_a = soc_eigs.get_orbital_magnetic_moments() @ easy_axis
    orbmag_sum = np.sum(orbmag_a)
    orbmag_max = np.max(np.abs(orbmag_a))

    results = {'orbmag_a': orbmag_a,
               'orbmag_sum': orbmag_sum,
               'orbmag_max': orbmag_max}

    return OrbMagResult(data=results)


if __name__ == '__main__':
    main.cli()
