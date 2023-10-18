"""Topological analysis of electronic structure."""
import numpy as np
from asr.core import command, option
from asr.paneldata import CalculateResult, BerryResult


@command(module='asr.berry',
         requires=['gs.gpw'],
         dependencies=['asr.gs'],
         resources='120:10h',
         returns=CalculateResult)
@option('--gs', help='Ground state', type=str)
@option('--kpar', help='K-points along path', type=int)
@option('--kperp', help='K-points orthogonal to path', type=int)
def calculate(gs: str = 'gs.gpw', kpar: int = 120,
              kperp: int = 7) -> CalculateResult:
    """Calculate ground state on specified k-point grid."""
    import os
    from ase.io import read
    from gpaw import GPAW
    from gpaw.berryphase import parallel_transport
    from gpaw.mpi import world
    from asr.magnetic_anisotropy import get_spin_axis

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)

    """Find the easy axis of magnetic materials"""
    theta, phi = get_spin_axis()

    results = {}
    results['phi0_km'] = None
    results['s0_km'] = None
    results['phi1_km'] = None
    results['s1_km'] = None
    results['phi2_km'] = None
    results['s2_km'] = None
    results['phi0_pi_km'] = None
    results['s0_pi_km'] = None

    if nd == 2:
        calc = GPAW(gs,
                    kpts=(kperp, kpar, 1),
                    fixdensity=True,
                    symmetry='off',
                    txt='gs_berry.txt')
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_km'] = phi_km
        results['s0_km'] = s_km

        if world.rank == 0:
            os.system('rm gs_berry.gpw')

    elif nd == 3:
        """kx = 0"""
        calc = GPAW(gs,
                    kpts=(1, kperp, kpar),
                    fixdensity=True,
                    symmetry='off',
                    txt='gs_berry.txt')
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=1,
                                          theta=theta,
                                          phi=phi)
        results['phi1_km'] = phi_km
        results['s1_km'] = s_km

        """ky = 0"""
        calc.set(kpts=(kpar, 1, kperp))
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=2,
                                          theta=theta,
                                          phi=phi)
        results['phi2_km'] = phi_km
        results['s2_km'] = s_km

        """kz = 0"""
        calc.set(kpts=(kperp, kpar, 1))
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_km'] = phi_km
        results['s0_km'] = s_km

        r"""kz = \pi"""
        from ase.dft.kpoints import monkhorst_pack
        kpts = monkhorst_pack((kperp, kpar, 1)) + [0, 0, 0.5]
        calc.set(kpts=kpts)
        calc.get_potential_energy()
        calc.write('gs_berry.gpw', mode='all')
        phi_km, s_km = parallel_transport('gs_berry.gpw',
                                          direction=0,
                                          theta=theta,
                                          phi=phi)
        results['phi0_pi_km'] = phi_km
        results['s0_pi_km'] = s_km

        if world.rank == 0:
            os.system('rm gs_berry.gpw')
    else:
        raise NotImplementedError('asr.berry@calculate is not implemented '
                                  'for <2D systems.')

    return CalculateResult(data=results)


@command(module='asr.berry',
         requires=['results-asr.berry@calculate.json'],
         dependencies=['asr.berry@calculate'],
         returns=BerryResult)
def main() -> BerryResult:
    from pathlib import Path
    from ase.parallel import paropen

    data = {}
    if Path('topology.dat').is_file():
        f = paropen('topology.dat', 'r')
        top = f.readline()
        f.close()
        data['Topology'] = top
    else:
        f = paropen('topology.dat', 'w')
        print('Not checked!', file=f)
        f.close()
        data['Topology'] = 'Not checked'

    return data


if __name__ == '__main__':
    main.cli()
