from asr.core import command, option, read_json

import numpy as np


def webpanel(row, key_descriptions):
    from asr.database.browser import fig, table

    opt = table(row, 'Property', [
        'alphax_lat', 'alphay_lat', 'alphaz_lat',
    ], key_descriptions)

    panel = {'title': 'Infrared polarizability (RPA)',
             'columns': [[fig('infrax.png'), fig('infraz.png')],
                         [fig('infray.png'), opt]],
             'plot_descriptions': [{'function': create_plot,
                                    'filenames': ['infrax.png',
                                                  'infray.png',
                                                  'infraz.png']}],
             'sort': 21}

    return [panel]


def create_plot(row, *fnames):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Get electronic polarizability
    infrareddct = row.data.get('results-asr.infraredpolarizability.json')
    omega_w = infrareddct['omega_w'] * 1e3
    alpha_wvv = infrareddct['alpha_wvv']

    electrondct = row.data.get('results-asr.polarizability.json')
    alphax_w = electrondct['alphax_w']
    alphay_w = electrondct['alphay_w']
    alphaz_w = electrondct['alphaz_w']
    omegatmp_w = electrondct['frequencies'] * 1e3

    # Get max phonon freq
    phonondata = row.data.get("results-asr.phonons.json")
    maxphononfreq = phonondata.get("omega_kl")[0].max() * 1e3
    maxomega = maxphononfreq * 1.5

    atoms = row.toatoms()
    pbc_c = atoms.pbc
    ndim = int(np.sum(pbc_c))

    realphax = interp1d(omegatmp_w, alphax_w.real)
    imalphax = interp1d(omegatmp_w, alphax_w.imag)
    ax_w = (realphax(omega_w) + 1j * imalphax(omega_w) +
            alpha_wvv[:, 0, 0])
    realphay = interp1d(omegatmp_w, alphay_w.real)
    imalphay = interp1d(omegatmp_w, alphay_w.imag)
    ay_w = (realphay(omega_w) + 1j * imalphay(omega_w) +
            alpha_wvv[:, 1, 1])
    realphaz = interp1d(omegatmp_w, alphaz_w.real)
    imalphaz = interp1d(omegatmp_w, alphaz_w.imag)
    az_w = (realphaz(omega_w) + 1j * imalphaz(omega_w) +
            alpha_wvv[:, 2, 2])

    if ndim == 3:
        epsx_w = 1 + 4 * np.pi * ax_w
        epsy_w = 1 + 4 * np.pi * ay_w
        epsz_w = 1 + 4 * np.pi * az_w
        plt.figure()
        plt.plot(omega_w, epsx_w.imag)
        plt.plot(omega_w, epsx_w.real)
        ax = plt.gca()
        ax.set_title('x-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(r'Dielectric function')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, epsy_w.imag)
        plt.plot(omega_w, epsy_w.real)
        ax = plt.gca()
        ax.set_title('y-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(r'Dielectric function')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, epsz_w.imag)
        plt.plot(omega_w, epsz_w.real)
        ax = plt.gca()
        ax.set_title('z-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(r'Dielectric function')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[2])
    elif ndim in [2, 1, 0]:
        if ndim == 2:
            unit = r'$\mathrm{\AA}$'
        elif ndim == 1:
            unit = r'$\mathrm{\AA}^2$'
        elif ndim == 0:
            unit = r'$\mathrm{\AA}^3$'
        plt.figure()
        plt.plot(omega_w, ax_w.imag)
        plt.plot(omega_w, ax_w.real)
        ax = plt.gca()
        ax.set_title('x-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, ay_w.imag)
        plt.plot(omega_w, ay_w.real)
        ax = plt.gca()
        ax.set_title('y-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, az_w.imag)
        plt.plot(omega_w, az_w.real)
        ax = plt.gca()
        ax.set_title('z-polarization')
        ax.set_xlabel('energy [meV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        ax.set_xlim(0, maxomega)
        plt.tight_layout()
        plt.savefig(fnames[2])


@command('asr.infraredpolarizability',
         dependencies=['asr.phonons', 'asr.borncharges', 'asr.polarizability'],
         requires=['structure.json', 'results-asr.phonons.json',
                   'results-asr.borncharges.json',
                   'results-asr.polarizability.json'],
         webpanel=webpanel)
@option('--nfreq', help='Number of frequency points')
@option('--eta', help='Relaxation rate')
def main(nfreq=300, eta=1e-2):
    from ase.io import read

    # Get relevant atomic structure
    atoms = read('structure.json')

    # Get phonons
    phresults = read_json('results-asr.phonons.json')
    u_ql = phresults['modes_kl']
    q_qc = phresults['q_qc']
    omega_ql = phresults['omega_kl']

    iq_q = np.argwhere((np.abs(q_qc) < 1e-10).all(axis=1))

    assert len(iq_q), 'Calculated phonons do not contain Gamma point.'

    iq = iq_q[0][0]

    m_a = atoms.get_masses()
    m_inv_x = np.repeat(m_a**-0.5, 3)
    freqs_l, modes_liv = omega_ql[iq], u_ql[iq]
    modes_xl = modes_liv.reshape(len(freqs_l), -1).T
    modes_xl *= 1 / m_inv_x[:, np.newaxis]

    # Make frequency grid
    fmin = 0
    fmax = omega_ql[0].max() * 3  # Factor of 3 should be enough
    omega_w = np.linspace(fmin, fmax, nfreq)

    # Read born charges
    borndct = read_json('results-asr.borncharges.json')

    # Get other relevant quantities
    m_a = atoms.get_masses()
    cell_cv = atoms.get_cell()
    Z_avv = borndct['Z_avv']

    # Get phonon polarizability
    alpha_wvv = get_phonon_pol(omega_w, Z_avv, freqs_l,
                               modes_xl, m_a, cell_cv, eta)

    # Normalize according to dimensionality
    pbc_c = atoms.pbc
    if pbc_c.all():
        norm = 1
    else:
        norm = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    alpha_wvv *= norm

    alphax_lat = alpha_wvv[0, 0, 0].real
    alphay_lat = alpha_wvv[0, 1, 1].real
    alphaz_lat = alpha_wvv[0, 2, 2].real

    elecdict = read_json('results-asr.polarizability.json')
    alphax_el = elecdict['alphax_el']
    alphay_el = elecdict['alphay_el']
    alphaz_el = elecdict['alphaz_el']

    results = {'alpha_wvv': alpha_wvv,
               'omega_w': omega_w,
               'alphax_lat': alphax_lat,
               'alphay_lat': alphay_lat,
               'alphaz_lat': alphaz_lat,
               'alphax': alphax_lat + alphax_el,
               'alphay': alphay_lat + alphay_el,
               'alphaz': alphaz_lat + alphaz_el}

    results['__key_descriptions__'] = {
        'alphax_lat': 'KVP: Static ionic polarizability, x-direction [Ang]',
        'alphay_lat': 'KVP: Static ionic polarizability, y-direction [Ang]',
        'alphaz_lat': 'KVP: Static ionic polarizability, z-direction [Ang]',
        'alphax': 'KVP: Static total polarizability, x-direction [Ang]',
        'alphay': 'KVP: Static total polarizability, y-direction [Ang]',
        'alphaz': 'KVP: Static total polarizability, z-direction [Ang]'}

    return results


def get_phonon_pol(omega_w, Z_avv, freqs_l, modes_xl, m_a, cell_cv, eta):
    from ase.units import Hartree, Bohr
    Z_vx = Z_avv.swapaxes(0, 1).reshape((3, -1))
    f2_w, D_xw = (freqs_l / Hartree)**2, modes_xl

    vol = abs(np.linalg.det(cell_cv)) / Bohr**3
    omega_w = omega_w / Hartree
    eta = eta / Hartree
    me = 1822.888
    m_a = m_a * me
    alpha_wvv = np.zeros((len(omega_w), 3, 3), dtype=complex)
    m_x = np.repeat(m_a, 3)**0.5
    eta = eta

    for f2, D_x in zip(f2_w, D_xw.T):
        # Neglect acoustic modes
        if f2 < (1e-3 / Hartree)**2:
            continue
        DM_x = D_x / m_x
        Z_v = np.dot(Z_vx, DM_x)
        alpha_wvv += (np.outer(Z_v, Z_v)[np.newaxis] /
                      ((f2 - omega_w**2) -
                       1j * eta * omega_w)[:, np.newaxis, np.newaxis])

    alpha_wvv /= vol
    return alpha_wvv


if __name__ == '__main__':
    main()
