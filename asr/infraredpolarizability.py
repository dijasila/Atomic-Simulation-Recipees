from asr.core import command, option, read_json

import numpy as np


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Infrared polarizability (RPA)',
             'columns': [[fig('infrax.png'), fig('infraz.png')],
                         [fig('infray.png')]],
             'plot_descriptions': [{'function': create_plot,
                                    'filenames': ['infrax.png',
                                                  'infray.png',
                                                  'infraz.png']}]}

    return [panel]


def create_plot(row, *fnames):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Get electronic polarizability
    infrareddct = row.data.get('results-asr.infraredpolarizability.json')
    omega_w = infrareddct['omega_w']
    alpha_wvv = infrareddct['alpha_wvv']

    electrondct = row.data.get('results-asr.polarizability.json')
    alphax_w = electrondct['alphax_w']
    alphay_w = electrondct['alphay_w']
    alphaz_w = electrondct['alphaz_w']
    omegatmp_w = electrondct['frequencies']

    atoms = row.toatoms()
    cell_cv = atoms.get_cell()
    pbc_c = atoms.pbc
    ndim = int(np.sum(pbc_c))
    if pbc_c.all():
        norm = 1
    else:
        norm = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

    alphax = interp1d(omegatmp_w, alphax_w)
    ax_w = (alphax(omega_w) + alpha_wvv[:, 0, 0] * norm)
    alphay = interp1d(omegatmp_w, alphay_w)
    ay_w = (alphay(omega_w) + alpha_wvv[:, 1, 1] * norm)
    alphaz = interp1d(omegatmp_w, alphaz_w)
    az_w = (alphaz(omega_w) + alpha_wvv[:, 2, 2] * norm)

    if ndim == 3:
        epsx_w = 1 + 4 * np.pi * ax_w
        epsy_w = 1 + 4 * np.pi * ay_w
        epsz_w = 1 + 4 * np.pi * az_w
        plt.figure()
        plt.plot(omega_w, epsx_w)
        ax = plt.gca()
        ax.set_title('x-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'Dielectric function')
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, epsy_w)
        ax = plt.gca()
        ax.set_title('y-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'Dielectric function')
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, epsz_w)
        ax = plt.gca()
        ax.set_title('z-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(r'Dielectric function')
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
        plt.plot(omega_w, ax_w)
        ax = plt.gca()
        ax.set_title('x-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        plt.tight_layout()
        plt.savefig(fnames[0])

        plt.figure()
        plt.plot(omega_w, ay_w)
        ax = plt.gca()
        ax.set_title('y-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        plt.tight_layout()
        plt.savefig(fnames[1])

        plt.figure()
        plt.plot(omega_w, az_w)
        ax = plt.gca()
        ax.set_title('z-component')
        ax.set_xlabel('energy [eV]')
        ax.set_ylabel(rf'polarizability [{unit}]')
        plt.tight_layout()
        plt.savefig(fnames[2])


@command('asr.infraredpolarizability',
         dependencies=['asr.phonons', 'asr.borncharges', 'asr.polarizability'],
         requires=['structure.json', 'results-asr.phonons.json',
                   'results-asr.borncharges.json',
                   'results-asr.polarizability.json'],
         webpanel=webpanel)
@option('--fmin', help='Minimum frequency')
@option('--fmax', help='Maximum frequency')
@option('--nfreq', help='Number of frequency points')
@option('--eta', help='Relaxation rate')
def main(fmin=0.0, fmax=1, nfreq=300, eta=1e-2):
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
