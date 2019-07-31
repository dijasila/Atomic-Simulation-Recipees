from asr.utils import command, option, read_json
from click import Choice


@command('asr.bse')
@option(
    '--gs', default='gs.gpw', help='Ground state on which BSE is based')
@option('--kptdensity', default=6.0, help='K-point density')
@option('--ecut', default=50.0, help='Plane wave cutoff')
@option('--nv', default=4, help='Valence bands included')
@option('--nc', default=4, help='Conduction bands included')
@option('--mode', default='BSE', help='Irreducible response',
        type=Choice(['RPA', 'BSE', 'TDHF']))
@option('--bandfactor', default=6, type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(gs, kptdensity, ecut, mode, bandfactor, nv, nc):
    """Calculate BSE polarizability"""
    import os
    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.bse import BSE
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np

    atoms = read('structure.json')
    pbc = atoms.pbc.tolist()

    ND = np.sum(pbc)
    if ND == 3:
        eta = 0.1
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        truncation = None
    elif ND == 2:
        eta = 0.05

        def get_kpts_size(atoms, kptdensity):
            """trying to get a reasonable monkhorst size which hits high
            symmetry points
            """
            from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
            size, offset = k2so(atoms=atoms, density=kptdensity)
            size[2] = 1
            for i in range(2):
                if size[i] % 6 != 0:
                    size[i] = 6 * (size[i] // 6 + 1)
            kpts = {'size': size, 'gamma': True}
            return kpts

        kpts = get_kpts_size(atoms=atoms, kptdensity=20)
        truncation = '2D'
        
    else:
        raise NotImplementedError(
            'asr for BSE not implemented for 0D and 1D structures')

    calc_old = GPAW(gs, txt=None)
    spin = calc_old.get_spin_polarized()
    nval = calc_old.wfs.nvalence
    nocc = int(nval / 2)
    nbands = bandfactor * nocc
    if not Path('gs_bse.gpw').is_file():
        calc = GPAW(
            gs,
            txt='gs_bse.txt',
            fixdensity=True,
            nbands=int(nbands * 1.5),
            convergence={'bands': nbands},
            occupations=FermiDirac(width=1e-4),
            kpts=kpts)
        calc.get_potential_energy()
        calc.write('gs_bse.gpw', mode='all')

    if spin:
        f0 = calc.get_occupation_numbers(spin=0)
        f1 = calc.get_occupation_numbers(spin=1)
        n0 = np.where(f0 < 1.0e-6)[0][0]
        n1 = np.where(f1 < 1.0e-6)[0][0]
        valence_bands = [range(n0 - nv, n0), range(n1 - nv, n1)]
        conduction_bands = [range(n0, n0 + nc), range(n1, n1 + nc)]
    else:
        valence_bands = range(nocc - nv, nocc)
        conduction_bands = range(nocc, nocc + nc)

    world.barrier()

    bse = BSE('gs_bse.gpw',
              spinors=True,
              ecut=ecut,
              valence_bands=valence_bands,
              conduction_bands=conduction_bands,
              nbands=nbands,
              mode=mode,
              wfile='wfile',
              truncation=truncation,
              txt='bse.txt')

    w_w = np.linspace(0.0, 5.0, 5001)

    w_w, alphax_w = bse.get_polarizability(eta=eta,
                                           filename=None,
                                           direction=0,
                                           write_eig='eig_x.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    w_w, alphay_w = bse.get_polarizability(eta=eta,
                                           filename=None,
                                           direction=1,
                                           write_eig='eig_y.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    w_w, alphaz_w = bse.get_polarizability(eta=eta,
                                           filename=None,
                                           direction=2,
                                           write_eig='eig_z.dat',
                                           pbc=pbc,
                                           w_w=w_w)

    eigx = np.loadtxt('eig_x.dat')
    eigy = np.loadtxt('eig_y.dat')
    eigz = np.loadtxt('eig_z.dat')

    if world.rank == 0:
        os.system('rm gs_bse.gpw')
        os.system('rm gs_nosym.gpw')
        os.system('rm wfile.npz')
        os.system('rm eig_x.dat')
        os.system('rm eig_y.dat')
        os.system('rm eig_z.dat')

    data = {
        'eigx': eigx,
        'eigy': eigy,
        'eigz': eigz,
        'alphax_w': alphax_w,
        'alphay_w': alphay_w,
        'alphaz_w': alphaz_w,
        'frequencies': w_w
    }

    return data


def collect_data(atoms):
    from pathlib import Path
    if not Path('results_bse.json').is_file():
        return {}, {}, {}

    kvp = {}
    data = {}
    key_descriptions = {}

    dct = read_json('results_bse.json')
    gap_dir = read_json('gap_soc.json')['gap_dir']
    kvp['bse_binding'] = gap_dir - dct['eigx'][0, 1]

    kd = {'bse_binding': ('BSE binding energy',
                          'BSE binding energy', 'eV')}
    key_descriptions.update(kd)

    print(kvp['bse_binding'])
    # Save data
    return kvp, key_descriptions, data


def polarizability(row, fx, fy, fz):
    import numpy as np
    import matplotlib.pyplot as plt

    def xlim():
        return (0, 10)

    def ylims(ws, data, wstart=0.0):
        i = abs(ws - wstart).argmin()
        x = data[i:]
        x1, x2 = x.real, x.imag
        y1 = min(x1.min(), x2.min()) * 1.02
        y2 = max(x1.max(), x2.max()) * 1.02
        return y1, y2

    if 'bse_polarization' in row.data:
        data = row.data['bse_polarization']
        frequencies = data['frequencies']
        alphax_w = data['alphax_w']
        alphay_w = data['alphay_w']
        alphaz_w = data['alphaz_w']

        ax = plt.figure().add_subplot(111)
        ax1 = ax
        try:
            wpx = row.plasmafrequency_x
            if wpx > 0.01:
                alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(
                    frequencies,
                    np.real(alphaxfull_w),
                    '-',
                    c='C1',
                    label='real')
                ax.plot(
                    frequencies,
                    np.real(alphax_w),
                    '--',
                    c='C1',
                    label='real interband')
            else:
                ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphax_w), c='C0', label='imag')
        ax.set_title('x-direction')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphax_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fx)

        ax = plt.figure().add_subplot(111)
        ax2 = ax
        try:
            wpy = row.plasmafrequency_y
            if wpy > 0.01:
                alphayfull_w = alphay_w - wpy**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(
                    frequencies,
                    np.real(alphayfull_w),
                    '-',
                    c='C1',
                    label='real')
                ax.plot(
                    frequencies,
                    np.real(alphay_w),
                    '--',
                    c='C1',
                    label='real interband')
            else:
                ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
        except AttributeError:
            ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphay_w), c='C0', label='imag')
        ax.set_title('y-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphax_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fy)

        ax = plt.figure().add_subplot(111)
        ax3 = ax
        ax.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
        ax.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
        ax.set_title('z-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alphaz_w, wstart=0.5))
        ax.legend()
        ax.set_xlim(xlim())
        plt.tight_layout()
        plt.savefig(fz)

        return ax1, ax2, ax3


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig, table

    opt = table(row, 'Property', [
        'alphax', 'alphay', 'alphaz', 'plasmafrequency_x', 'plasmafrequency_y'
    ], key_descriptions)

    panel = ('Polarizability (RPA)',
             [[fig('rpa-pol-x.png'),
               fig('rpa-pol-z.png')], [fig('rpa-pol-y.png'), opt]])

    things = [(polarizability,
               ['rpa-pol-x.png', 'rpa-pol-y.png', 'rpa-pol-z.png'])]

    return panel, things


group = 'property'
dependencies = ['asr.structureinfo', 'asr.gs', 'asr.gaps']

if __name__ == '__main__':
    main()
