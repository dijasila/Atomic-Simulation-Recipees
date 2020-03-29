import numpy as np
from asr.core import command, option

@command(module='asr.exchange',
         creates=['2x1x1_gs.gpw', 'exchange.gpw'],
         requires=['gs.gpw'],
         resources='40:10h')
@option('--gs', help='Ground state on which exchange calculation is based')
def calculate(gs='gs.gpw'):
    """Calculate two spin configurations"""
    from gpaw import GPAW
    from asr.core import magnetic_atoms

    calc = GPAW(gs, fixdensity=False, txt=None)
    atoms = calc.atoms
    magnetic = magnetic_atoms(atoms)
    if sum(magnetic) == 2:
        calc.reset()
        calc.set(txt='2x1x1_gs.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('2x1x1_gs.gpw')

        a1, a2 = np.where(magnetic)[0]
        magmoms_i = calc.get_magnetic_moments()
        magmoms_e = np.zeros(len(atoms), float)
        magmoms_e[a1] = np.max(magmoms_i)
        if np.sign(magmoms_i[a1]) == np.sign(magmoms_i[a2]):
            magmoms_e[a2] = -np.max(magmoms_i)
        else:
            magmoms_e[a2] = np.max(magmoms_i)
        atoms.set_initial_magnetic_moments(magmoms_e)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('exchange.gpw')

    elif sum(magnetic) == 1:
        a1 = np.where(magnetic)[0]
        mag = np.max(calc.get_magnetic_moments())
        magmoms = np.zeros(len(atoms), float)
        magmoms[a1] = mag
        atoms.set_initial_magnetic_moments(magmoms)
        atoms = atoms.repeat((2, 1, 1))
        calc.reset()
        calc.set(txt='2x1x1_gs.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('2x1x1_gs.gpw')

        magnetic = magnetic_atoms(atoms)
        a1, a2 = np.where(magnetic)[0]
        mag = np.max(calc.get_magnetic_moments())
        magmoms = np.zeros(len(atoms), float)
        magmoms[a1] = mag
        magmoms[a2] = -mag
        atoms.set_initial_magnetic_moments(magmoms)
        calc.reset()
        calc.set(txt='exchange.txt')
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('exchange.gpw')

    else:
        pass

def get_parameters(fm, afm, txt=False, 
                   dis_cut=0.2, line=False, a0=None):
    """Extract Heisenberg parameters"""
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from gpaw.spinorbit import get_anisotropy
    from ase.geometry import get_distances
    from ase.dft.bandgap import bandgap

    calc_fm = GPAW(fm, communicator=serial_comm, txt=None)
    calc_afm = GPAW(afm, communicator=serial_comm, txt=None)
    nbands = calc_afm.get_number_of_bands()
    atoms = calc_fm.atoms
    if a0 is None:
        a0 = np.argmax(np.abs(calc_fm.get_magnetic_moments()))
    symbols = atoms.symbols
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

    E_fm_x = get_anisotropy(calc_fm, theta=np.pi/2, phi=0,
                            width=width, nbands=nbands) / 2
    E_fm_y = get_anisotropy(calc_fm, theta=np.pi/2, phi=np.pi/2,
                            width=width, nbands=nbands) / 2
    E_fm_z = get_anisotropy(calc_fm, theta=0, phi=0,
                            width=width, nbands=nbands) / 2
    E_afm_x = get_anisotropy(calc_afm, theta=np.pi/2, phi=0,
                             width=width, nbands=nbands) / 2
    E_afm_y = get_anisotropy(calc_afm, theta=np.pi/2, phi=np.pi/2,
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
         requires=['2x1x1_gs.gpw', 'exchange.gpw'],
         dependencies=['asr.exchange@calculate', 'asr.gs'])
def main():
    """Collect data"""
    from ase.io import read
    N_gs = len(read('gs.gpw'))
    N_exchange = len(read('2x1x1_gs.gpw'))
    if N_gs == N_exchange:
        line = False
    else:
        line = True

    J, A, B = get_parameters('2x1x1_gs.gpw', 'exchange.gpw', line=line)
    
    data = {}
    data['J'] = J
    data['__key_descriptions__'] = {'J': 'KVP: Exchange coupling [meV]'}
    data['A'] = A
    data['__key_descriptions__'] = {'A': 'KVP: Single-ion anisotropy [meV]'}
    data['B'] = B
    data['__key_descriptions__'] = {'B': 'KVP: Anisotropic exchange [meV]'}
    return data


if __name__ == '__main__':
    main.cli()
