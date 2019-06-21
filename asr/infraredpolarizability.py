import json
from pathlib import Path
from asr.utils import command, option, read_json

import numpy as np

from ase.units import Hartree, Bohr
from ase.io import read

@command('asr.infraredpolarizability')
@option('--frequencies', default=[0.0, 0.4, 100], help='frequency grid')
@option('--eta', default=1e-4, help='relaxation rate')
def main(frequencies, eta):
    # Get relevant atomic structure                                                                                                
    atoms = read('gs.gpw')

    # Get phonons
    from asr.phonons import analyse
    q_qc = [[0, 0, 0], ]
    omega_ql, u_ql, q_qc = analyse(modes=True, q_qc = q_qc)

    freqs_l, modes_liv = omega_ql[0] / Hartree, u_ql[0]
    modes_xl = modes_liv.reshape(len(freqs_l), -1).T
    modes_xl *= (analyse.dct['phonons']['m_inv_x']**(-1))[:, np.newaxis]

    # Make frequency grid
    omega_w = np.linspace(*frequencies) / Hartree

    # Read born charges
    bornchargefile = '/data-borncharges/borncharges-0.01.json'
    borndct = read_json(bornchargefile)

    # Get other relevant quantities
    me = 1822.888
    m_a = atoms.get_masses() * me
    cell_cv = atoms.get_cell() / Bohr
    Z_avv = borndct['Z_avv']

    # Get phonon polarizability
    alpha_wvv = get_phonon_pol(omega_w, Z_avv, freqs_l,
                               modes_xl, m_a, cell_cv, eta)

    # Get electronic polarizability
    eletronicfile = 'polarizability.json'
    electrondct = read_json(electronicfile)
    alphax_w = electrondct['alphax_w']
    alphay_w = electrondct['alphay_w']
    alphaz_w = electrondct['alphaz_w']
    omegatmp_w = electrondct['frequencies'] / Hartree
    from scipy.interpolate import interp1d

    alphax = interp1d(omegatmp_w, alphax_w)
    ax = alphax(omega_w) + alpha_wvv[:, 0, 0]
    epsx_w = 1 + 4 * np.pi * ax

    alphay = interp1d(omegatmp_w, alphay_w)
    ay = alphay(omega_w) + alpha_wvv[:, 1, 1]
    epsy_w = 1 + 4 * np.pi * ay
    
    alphaz = interp1d(omegatmp_w, alphaz_w)
    az = alphaz(omega_w) + alpha_wvv[:, 2, 2]
    epsz_w = 1 + 4 * np.pi * az

    results = {'alphax': ax,
               'epsx': epsx_w,
               'alphay': ay,
               'epsy': epsy_w,
               'alphaz': az,
               'epsz': epsz_w,
               'omega': omega_w * Hartree}

    return results


def get_phonon_pol(omega_w, Z_avv, freqs_l, modes_xl, m_a, cell_cv, eta):

    # Get phonons at q=0
    Z_vx = Z_avv.swapaxes(0, 1).reshape((3, -1))
    f2_w, D_xw = (freqs_l / Hartree)**2, modes_xl

    alpha_wvv = np.zeros((len(omega_w), 3, 3), dtype=complex)
    m_x = np.repeat(m_a, 3)**0.5
    eta = eta / Hartree

    for f2, D_x in zip(f2_w, D_xw.T):
        if f2 < (1e-3 / Hartree)**2:
            continue
        DM_x = D_x / m_x
        Z_v = np.dot(Z_vx, DM_x)
        alpha_wvv += (np.outer(Z_v, Z_v)[np.newaxis] / 
                      ((f2 - omega_w**2) - 
                       1j * eta * omega_w)[:, np.newaxis, np.newaxis])

    vol = abs(np.linalg.det(cell_cv)) / Bohr**3
    L = np.abs(cell_cv[2, 2] / Bohr)
    alpha_wvv *= 1 / vol * L
    
    return alpha_wvv

# def collect_data(atoms):
#     path = Path('results_infraredpolarizability.json')
#     if not path.is_file():
#         return {}, {}, {}

#     dct = json.loads(path.read_text())
#     kvp = {'alphax': dct['alphax'][0].real,
#            'alphay': dct['alphay'][0].real,
#            'alphaz': dct['alphaz'][0].real}
#     kd = {'alphax': ('Infrared static polarizability (x-direction)',
#                      'Infrared static polarizability (x-direction)', 'Ang'),
#           'alphay': ('Infrared static polarizability (y-direction)',
#                      'Infrared static polarizability (y-direction)', 'Ang'),
#           'alphaz': ('Infrared static polarizability (z-direction)',
#                      'Infrared static polarizability (z-direction)', 'Ang')}
#     data = {'infraredpolarizability': dct}
#     return kvp, kd, data


# def webpanel(row, key_descriptions):
#     from asr.utils.custom import fig, table

#     if 'something' not in row.data:
#         return None, []

#     table1 = table(row,
#                    'Property',
#                    ['something'],
#                    kd=key_descriptions)
#     panel = ('Title',
#              [[fig('something.png'), table1]])
#     things = [(create_plot, ['something.png'])]
#     return panel, things


# def create_plot(row, fname):
#     import matplotlib.pyplot as plt

#     freqs = row.data.get()
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.plot(data.things)
#     plt.savefig(fname)


group = 'property'
creates = ['results_infraredpolarizability.json']  # what files are created
dependencies = ['asr.phonons', 'asr.borncharges', 'asr.polarizability'] 
resources = '1:10m'  # 1 core for 10 minutes

if __name__ == '__main__':
    main()
