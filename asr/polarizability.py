from asr.utils import command, option


@command()
@option(
    '--gs', default='gs.gpw', help='Ground state on which response is based')
@option('--density', default=20.0, help='K-point density')
@option('--ecut', default=50.0, help='Plane wave cutoff')
@option('--xc', default='RPA', help='XC interaction (RPA or ALDA)')
def main(gs, density, ecut, xc):
    """Calculate linear response polarizability or dielectricfunction
    (only in 3D)"""
    import json
    from ase.io import jsonio
    from gpaw import GPAW
    from gpaw.mpi import world
    from gpaw.response.df import DielectricFunction
    from gpaw.occupations import FermiDirac
    from pathlib import Path
    import numpy as np

    from asr.utils import get_start_atoms
    atoms = get_start_atoms()
    pbc = atoms.pbc.tolist()

    dfkwargs = {
        'eta': 0.05,
        'domega0': 0.005,
        'ecut': ecut,
        'name': 'chi',
        'intraband': False
    }

    ND = np.sum(pbc)
    if ND == 3:
        kpts = {'density': density, 'gamma': False, 'even': True}
    elif ND == 2:

        def get_kpts_size(atoms, density):
            """trying to get a reasonable monkhorst size which hits high
            symmetry points
            """
            from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
            size, offset = k2so(atoms=atoms, density=density)
            size[2] = 1
            for i in range(2):
                if size[i] % 6 != 0:
                    size[i] = 6 * (size[i] // 6 + 1)
            kpts = {'size': size, 'gamma': True}
            return kpts

        kpts = get_kpts_size(atoms=atoms, density=density)
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
            nbands=6 * nval,
            convergence={'bands': 5 * nval},
            occupations=FermiDirac(width=1e-4),
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

    filename = 'polarizability.json'

    if world.rank == 0:
        Path(filename).write_text(json.dumps(data, cls=jsonio.MyEncoder))


def collect_data(atoms):
    from pathlib import Path
    import json
    from ase.io import jsonio
    if not Path('polarizability.json').is_file():
        return {}, {}, {}

    kvp = {}
    data = {}
    key_descriptions = {}
    dct = json.loads(Path('polarizability.json').read_text(),
                     cls=jsonio.MyEncoder)

    # Update key-value-pairs
    kvp['alphax'] = dct['alphax_w'][0].real
    kvp['alphay'] = dct['alphay_w'][0].real
    kvp['alphaz'] = dct['alphaz_w'][0].real

    # Update key_descriptions
    kd = {
        'alphax': ('Static polarizability (x-direction)',
                   'Static polarizability (x-direction)', 'Ang'),
        'alphay': ('Static polarizability (y-direction)',
                   'Static polarizability (y-direction)', 'Ang'),
        'alphaz': ('Static polarizability (z-direction)',
                   'Static polarizability (z-direction)', 'Ang')
    }
    key_descriptions.update(kd)

    # Save data
    data['absorptionspectrum'] = dct
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

    if 'absorptionspectrum' in row.data:
        data = row.data['absorptionspectrum']
        frequencies = data['frequencies']
        i2 = abs(frequencies - 10.0).argmin()
        frequencies = frequencies[:i2]
        alphax_w = data['alphax_w'][:i2]
        alphay_w = data['alphay_w'][:i2]
        alphaz_w = data['alphaz_w'][:i2]

        ax = plt.figure().add_subplot(111)
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
        plt.close()

        ax = plt.figure().add_subplot(111)
        try:
            wpy = row.plasmafrequency_y
            if wpy > 0.01:
                alphayfull_w = alphay_w - wpy**2 / (2 * np.pi *
                                                    (frequencies + 1e-9)**2)
                ax.plot(
                    frequencies,
                    np.real(alphayfull_w),
                    '--',
                    c='C1',
                    label='real')
                ax.plot(
                    frequencies,
                    np.real(alphay_w),
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
        plt.close()

        ax = plt.figure().add_subplot(111)
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
        plt.close()


def webpanel(row, key_descriptions):
    from asr.custom import fig, table

    opt = table(row, 'Property', [
        'alphax', 'alphay', 'alphaz', 'plasmafrequency_x', 'plasmafrequency_y'
    ], key_descriptions)

    panel = ('Polarizability (RPA)',
             [[fig('rpa-pol-x.png'),
               fig('rpa-pol-z.png')], [fig('rpa-pol-y.png'), opt]])

    things = [(polarizability,
               ['rpa-pol-x.png', 'rpa-pol-y.png', 'rpa-pol-z.png'])]

    return panel, things


group = 'Property'
creates = ['polarizability.npz', 'chi+0+0+0.pckl']
dependencies = ['asr.gs']

if __name__ == '__main__':
    main()
