import numpy as np
from asr.core import command, option, read_json, ASRResult


@command(module='asr.berry',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         resources='120:10h')
@option('--gs', help='Ground state', type=str)
@option('--kpar', help='K-points along path', type=int)
@option('--kperp', help='K-points orthogonal to path', type=int)
def calculate(gs: str = 'gs.gpw', kpar: int = 120, kperp: int = 7):
    """Calculate ground state on specified k-point grid."""
    import os
    from ase.io import read
    from gpaw import GPAW
    from gpaw.berryphase import parallel_transport
    from gpaw.mpi import world

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()

    """Find the easy axis of magnetic materials"""
    theta = 0.0
    phi = 0.0
    if os.path.isfile('results-asr.magnetic_anisotropy.json'):
        a = read_json('results-asr.magnetic_anisotropy.json')
        dE_zy = a['dE_zy']
        dE_zx = a['dE_zx']
        if dE_zy > 0 or dE_zx > 0:
            theta = np.pi / 2
            if dE_zy > dE_zx:
                phi = np.pi / 2

    ND = np.sum(pbc)

    results = {}

    if ND == 2:
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

        return results

    elif ND == 3:
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

        return results

    else:
        return


def plot_phases(name='0', fname='berry', show=False):
    import pylab as plt

    results = read_json('results-asr.berry@calculate.json')
    phit_km = results['phi%s_km' % name]
    St_km = results['s%s_km' % name]
    Nk = len(St_km)

    phi_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
    phi_km[1:] = phit_km
    phi_km[0] = phit_km[-1]
    S_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
    S_km[1:] = St_km
    S_km[0] = St_km[-1]
    S_km /= 2

    Nm = len(phi_km[0])
    phi_km = np.tile(phi_km, (1, 2))
    phi_km[:, Nm:] += 2 * np.pi
    S_km = np.tile(S_km, (1, 2))

    plt.figure()
    plt.scatter(np.tile(np.arange(len(phi_km)), len(phi_km.T)),
                phi_km.T.reshape(-1),
                cmap=plt.get_cmap('viridis'),
                c=S_km.T.reshape(-1),
                s=5,
                marker='o')

    cbar = plt.colorbar()
    cbar.set_label(r'$\langle S_z\rangle/\hbar$', size=20)

    plt.ylabel(r'$\gamma_x$', size=24)
    plt.xticks([0, Nk / 2, Nk],
               [r'$-\mathrm{M}$', r'$\Gamma$', r'$\mathrm{M}$'], size=20)
    plt.yticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], size=20)
    plt.axis([0, Nk, 0, 2 * np.pi])
    plt.tight_layout()
    figname = f'{fname}-phi{name}.png'
    plt.savefig(figname)
    if show:
        plt.show()


def webpanel(result, row, key_descriptions):
    if row.Topology == 'Not checked':
        return []

    row = ['Band topology', row.Topology]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Electronic properties', ''],
                             'rows': [row]}]]}

    basicelec = {'title': 'Basic electronic properties (PBE)',
                 'columns': [[{'type': 'table',
                               'header': ['Property', ''],
                               'rows': [row]}]],
                 'sort': 15}

    return [summary, basicelec]


class Result(ASRResult):

    formats = {"ase_webpanel": webpanel}


@command(module='asr.berry',
         requires=['results-asr.berry@calculate.json'],
         dependencies=['asr.berry@calculate'],
         returns=Result)
def main():
    from pathlib import Path
    from ase.parallel import paropen

    data = {}
    if Path('topology.dat').is_file():
        f = paropen('topology.dat', 'r')
        top = f.readline()
        f.close()
        data['Topology'] = top
        data['__key_descriptions__'] = \
            {'Topology': 'KVP: Band topology (Topology)'}
    else:
        f = paropen('topology.dat', 'w')
        print('Not checked!', file=f)
        f.close()
        data['Topology'] = 'Not checked'

    return data


if __name__ == '__main__':
    main.cli()
