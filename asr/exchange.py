import numpy as np
from asr.core import command, option, file_barrier


@command(module='asr.exchange',
         creates=['gs_2mag.gpw', 'exchange.gpw'],
         requires=['gs.gpw'],
         resources='40:10h')
@option('--gs', help='Ground state on which exchange calculation is based')
def calculate(gs='gs.gpw'):
    """Calculate two spin configurations."""
    from gpaw import GPAW
    from asr.core import magnetic_atoms

    calc = GPAW(gs, fixdensity=False, txt=None)
    atoms = calc.atoms
    magnetic = magnetic_atoms(atoms)
    assert sum(magnetic) in [1, 2], \
        ('Cannot handle %d magnetic atoms' % sum(magnetic))
    if sum(magnetic) == 2:
        calc.reset()
        calc.set(txt='gs_2mag.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('gs_2mag.gpw')

        a1, a2 = np.where(magnetic)[0]
        magmoms_i = calc.get_magnetic_moments()
        assert np.round(np.abs(magmoms_i[a1] / magmoms_i[a2]), 1) == 1, \
            'The two magnetic moments differ'
        magmoms_e = np.zeros(len(atoms), float)
        magmoms_e[a1] = np.max(np.abs(magmoms_i))
        if np.sign(magmoms_i[a1]) == np.sign(magmoms_i[a2]):
            magmoms_e[a2] = -magmoms_e[a1]
        else:
            magmoms_e[a2] = magmoms_e[a1]
        atoms.set_initial_magnetic_moments(magmoms_e)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('exchange.gpw')

    else:
        a1 = np.where(magnetic)[0]
        mag = np.max(np.abs(calc.get_magnetic_moments()))
        magmoms = np.zeros(len(atoms), float)
        magmoms[a1] = mag
        atoms.set_initial_magnetic_moments(magmoms)
        atoms = atoms.repeat((2, 1, 1))
        calc.reset()
        calc.set(txt='gs_2mag.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('gs_2mag.gpw')

        magnetic = magnetic_atoms(atoms)
        a1, a2 = np.where(magnetic)[0]
        magmoms_i = calc.get_magnetic_moments()
        assert np.round(magmoms_i[a1] / magmoms_i[a2], 1) == 1, \
            'The two magnetic moments differ'
        mag = np.max(np.abs(magmoms_i))
        magmoms_e = np.zeros(len(atoms), float)
        magmoms_e[a1] = mag
        magmoms_e[a2] = -mag
        atoms.set_initial_magnetic_moments(magmoms_e)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('exchange.gpw')


def get_parameters(gs, exchange, txt=False,
                   dis_cut=0.2, line=False, a0=None):
    """Extract Heisenberg parameters."""
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from gpaw.spinorbit import get_anisotropy
    from ase.dft.bandgap import bandgap
    from gpaw.utilities.ibz2bz import ibz2bz

    with file_barrier(['gs_2mag_nosym.gpw']):
        ibz2bz(gs, 'gs_2mag_nosym.gpw')
    with file_barrier(['exchange_nosym.gpw']):
        ibz2bz(exchange, 'exchange_nosym.gpw')

    calc_gs_2mag = GPAW('gs_2mag_nosym.gpw', communicator=serial_comm, txt=None)
    calc_exchange = GPAW('exchange_nosym.gpw', communicator=serial_comm, txt=None)
    m_gs = calc_gs_2mag.get_magnetic_moment()
    m_ex = calc_exchange.get_magnetic_moment()
    if np.abs(m_gs) > np.abs(m_ex):
        assert np.abs(m_gs) - np.abs(m_ex) > 0.1, \
            'AFM calculation did not converge to target state'
        calc_fm = calc_gs_2mag
        calc_afm = calc_exchange
    else:
        assert np.abs(m_ex) - np.abs(m_gs) > 0.1, \
            'AFM calculation did not converge to target state'
        calc_afm = calc_gs_2mag
        calc_fm = calc_exchange

    nbands = calc_afm.get_number_of_bands()
    atoms = calc_fm.atoms
    if a0 is None:
        a0 = np.argmax(np.abs(calc_fm.get_magnetic_moments()))
    el = atoms[a0].symbol
    a_i = []
    for i in range(len(atoms)):
        if atoms[i].symbol == el:
            a_i.append(i)
    atoms = atoms[a_i].repeat((3, 3, 1))
    dis_i = atoms.get_distances(a0, range(len(atoms)), mic=True)
    dis0 = np.sort(dis_i)[1]
    N = len(np.where(np.sort(dis_i)[1:] / dis0 - 1 < dis_cut)[0])

    E_fm = calc_fm.get_potential_energy() / 2
    E_afm = calc_afm.get_potential_energy() / 2

    gap_fm, p1, p2 = bandgap(calc_fm, output=None)
    gap_afm, p1, p2 = bandgap(calc_afm, output=None)
    if gap_fm > 0 and gap_afm > 0:
        width = 0.001
    else:
        width = None

    E_fm_x = get_anisotropy(calc_fm, theta=np.pi / 2, phi=0,
                            width=width, nbands=nbands) / 2
    E_fm_y = get_anisotropy(calc_fm, theta=np.pi / 2, phi=np.pi / 2,
                            width=width, nbands=nbands) / 2
    E_fm_z = get_anisotropy(calc_fm, theta=0, phi=0,
                            width=width, nbands=nbands) / 2
    E_afm_x = get_anisotropy(calc_afm, theta=np.pi / 2, phi=0,
                             width=width, nbands=nbands) / 2
    E_afm_y = get_anisotropy(calc_afm, theta=np.pi / 2, phi=np.pi / 2,
                             width=width, nbands=nbands) / 2
    E_afm_z = get_anisotropy(calc_afm, theta=0, phi=0,
                             width=width, nbands=nbands) / 2
    E_fm_x = (E_fm_x + E_fm_y) / 2
    E_afm_x = (E_afm_x + E_afm_y) / 2

    dE_fm = (E_fm_x - E_fm_z)
    dE_afm = (E_afm_x - E_afm_z)

    S = np.abs(np.round(calc_fm.get_magnetic_moment() / 2))
    S = S / 2
    if S == 0:
        S = 1 / 2

    if line:
        if N == 4:
            N_afm = 2
            N_fm = 2
        elif N == 6:
            N_afm = 4
            N_fm = 2
        else:
            print('Line not recognized')
            N_afm = N
            N_fm = 0
    else:
        if N == 6:
            N_afm = 4
            N_fm = 2
        elif N == 9:
            N_afm = 3
            N_fm = 6
        else:
            N_afm = N
            N_fm = 0

    J = (E_afm + E_afm_x - E_fm - E_fm_x) / (N_afm * S**2)

    if np.abs(S - 0.5) < 0.1:
        A = 0
        if J > 0 or (N == 4 and line):
            B = dE_fm / (N * S**2)
        else:
            B = -dE_afm / (N_afm - N_fm) / S**2
    else:
        A = dE_fm * (1 - N_fm / N_afm) + dE_afm * (1 + N_fm / N_afm)
        A /= (2 * S - 1) * S
        B = (dE_fm - dE_afm) / (N_afm * S**2)

    return J, A, B


@command(module='asr.exchange',
         requires=['gs_2mag.gpw', 'exchange.gpw'],
         dependencies=['asr.exchange@calculate', 'asr.gs'])
def main():
    """Collect data."""
    from ase.io import read
    N_gs = len(read('gs.gpw'))
    N_exchange = len(read('gs_2mag.gpw'))
    if N_gs == N_exchange:
        line = False
    else:
        line = True

    J, A, B = get_parameters('gs_2mag.gpw', 'exchange.gpw', line=line)

    data = {}
    data['J'] = J * 1000
    data['__key_descriptions__'] = {'J': 'KVP: Exchange coupling [meV]'}
    data['A'] = A * 1000
    data['__key_descriptions__'] = {'A': 'KVP: Single-ion anisotropy [meV]'}
    data['B'] = B * 1000
    data['__key_descriptions__'] = {'B': 'KVP: Anisotropic exchange [meV]'}
    return data


if __name__ == '__main__':
    main.cli()
