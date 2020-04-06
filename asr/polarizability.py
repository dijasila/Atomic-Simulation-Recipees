from asr.core import command, option
from click import Choice


def webpanel(row, key_descriptions):
    from asr.database.browser import fig, table

    opt = table(row, 'Property', [
        'alphax_el', 'alphay_el', 'alphaz_el',
        'alphax_lat', 'alphay_lat', 'alphaz_lat',
        'alphax', 'alphay', 'alphaz',
    ], key_descriptions)

    panel = {'title': 'Optical polarizability (RPA)',
             'columns': [[fig('rpa-pol-x.png'), fig('rpa-pol-z.png')],
                         [fig('rpa-pol-y.png'), opt]],
             'plot_descriptions':
                 [{'function': polarizability,
                   'filenames': ['rpa-pol-x.png',
                                 'rpa-pol-y.png',
                                 'rpa-pol-z.png']}],
             'sort': 20}

    return [panel]


def get_kpts_size(atoms, density):
    """Try to get a reasonable monkhorst size which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=density)
    size[2] = 1
    for i in range(2):
        if size[i] % 6 != 0:
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts


@command('asr.polarizability',
         dependencies=['asr.structureinfo', 'asr.gs@calculate'],
         requires=['gs.gpw'],
         webpanel=webpanel)
@option(
    '--gs', help='Ground state on which response is based')
@option('--kptdensity', help='K-point density')
@option('--ecut', help='Plane wave cutoff')
@option('--xc', help='XC interaction', type=Choice(['RPA', 'ALDA']))
@option('--bandfactor', type=int,
        help='Number of unoccupied bands = (#occ. bands) * bandfactor)')
def main(gs='gs.gpw', kptdensity=20.0, ecut=50.0, xc='RPA', bandfactor=5):
    """Calculate linear response polarizability or dielectricfunction (only in 3D)."""
    from ase.io import read
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.df import DielectricFunction
    from pathlib import Path
    import numpy as np

    try:
        atoms = read('structure.json')
        pbc = atoms.pbc.tolist()

        dfkwargs = {
            'eta': 0.05,
            'domega0': 0.005,
            'ecut': ecut,
            'name': 'chi',
            'intraband': False
        }

        ND = np.sum(pbc)
        if ND == 3 or ND == 1:
            kpts = {'density': kptdensity, 'gamma': False, 'even': True}
        elif ND == 2:
            kpts = get_kpts_size(atoms=atoms, density=kptdensity)
            volume = atoms.get_volume()
            if volume < 120:
                nblocks = world.size // 4
            else:
                nblocks = world.size // 2
            dfkwargs.update({
                'nblocks': nblocks,
                'pbc': pbc,
                'integrationmode': 'tetrahedron integration',
                'truncation': '2D'
            })

        else:
            raise NotImplementedError(
                'Polarizability not implemented for 1D and 2D structures')

        if not Path('es.gpw').is_file():
            calc_old = GPAW(gs, txt=None)
            nval = calc_old.wfs.nvalence

            calc = GPAW(
                gs,
                txt='es.txt',
                fixdensity=True,
                nbands=(bandfactor + 1) * nval,
                convergence={'bands': bandfactor * nval},
                occupations={'name': 'fermi-dirac',
                             'width': 1e-4},
                kpts=kpts)
            calc.get_potential_energy()
            calc.write('es.gpw', mode='all')

        df = DielectricFunction('es.gpw', **dfkwargs)
        alpha0x, alphax = df.get_polarizability(
            q_c=[0, 0, 0], direction='x', pbc=pbc, filename=None,
            xc=xc)
        alpha0y, alphay = df.get_polarizability(
            q_c=[0, 0, 0], direction='y', pbc=pbc, filename=None,
            xc=xc)
        alpha0z, alphaz = df.get_polarizability(
            q_c=[0, 0, 0], direction='z', pbc=pbc, filename=None,
            xc=xc)

        plasmafreq_vv = df.chi0.plasmafreq_vv

        frequencies = df.get_frequencies()
        data = {
            'alpha0x_w': np.array(alpha0x),
            'alphax_w': np.array(alphax),
            'alpha0y_w': np.array(alpha0y),
            'alphay_w': np.array(alphay),
            'alpha0z_w': np.array(alpha0z),
            'alphaz_w': np.array(alphaz),
            'plasmafreq_vv': plasmafreq_vv,
            'frequencies': frequencies
        }

        data['alphax_el'] = data['alphax_w'][0].real
        data['alphay_el'] = data['alphay_w'][0].real
        data['alphaz_el'] = data['alphaz_w'][0].real

        data['__key_descriptions__'] = {
            'alphax_el': 'KVP: Static electronic polarizability,'
            ' x-direction [Ang]',
            'alphay_el': 'KVP: Static electronic polarizability,'
            ' y-direction [Ang]',
            'alphaz_el': 'KVP: Static electronic polarizability,'
            ' z-direction [Ang]'}

        world.barrier()
        if world.rank == 0:
            es_file = Path("es.gpw")
            es_file.unlink()

    except MemoryError as error:
        if world.rank == 0:
            es_file = Path("es.gpw")
            es_file.unlink()
        raise error

    return data


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

    data = row.data.get('results-asr.polarizability.json')

    if data is None:
        return
    frequencies = data['frequencies']
    i2 = abs(frequencies - 50.0).argmin()
    frequencies = frequencies[:i2]
    alphax_w = data['alphax_w'][:i2]
    alphay_w = data['alphay_w'][:i2]
    alphaz_w = data['alphaz_w'][:i2]

    infrared = row.data.get('results-asr.infraredpolarizability.json')
    if infrared:
        from scipy.interpolate import interp1d
        omegatmp_w = infrared['omega_w']
        alpha_wvv = infrared['alpha_wvv']
        alphax = interp1d(omegatmp_w, alpha_wvv[:, 0, 0], fill_value=0,
                          bounds_error=False)
        alphax_w = (alphax_w + alphax(frequencies))
        alphay = interp1d(omegatmp_w, alpha_wvv[:, 1, 1], fill_value=0,
                          bounds_error=False)
        alphay_w = (alphay_w + alphay(frequencies))
        alphaz = interp1d(omegatmp_w, alpha_wvv[:, 2, 2], fill_value=0,
                          bounds_error=False)
        alphaz_w = (alphaz_w + alphaz(frequencies))

    ax = plt.figure().add_subplot(111)
    ax1 = ax
    try:
        wpx = row.plasmafrequency_x
        if wpx > 0.01:
            alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi * (frequencies + 1e-9)**2)
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
    ax.set_title('x-polarization')
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
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
            alphayfull_w = alphay_w - wpy**2 / (2 * np.pi * (frequencies + 1e-9)**2)
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
    ax.set_title('y-polarization')
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
    ax.set_ylim(ylims(ws=frequencies, data=alphax_w, wstart=0.5))
    ax.legend()
    ax.set_xlim(xlim())
    plt.tight_layout()
    plt.savefig(fy)

    ax = plt.figure().add_subplot(111)
    ax3 = ax
    ax.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
    ax.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
    ax.set_title('z-polarization')
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
    ax.set_ylim(ylims(ws=frequencies, data=alphaz_w, wstart=0.5))
    ax.legend()
    ax.set_xlim(xlim())
    plt.tight_layout()
    plt.savefig(fz)

    return ax1, ax2, ax3


if __name__ == '__main__':
    main.cli()
