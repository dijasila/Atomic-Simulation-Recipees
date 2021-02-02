"""Self-consistent EF calculation for defect systems.."""
from asr.core import command, option, DictStr, ASRResult, prepare_result
from ase.dft.bandgap import bandgap
from gpaw import restart
import numpy as np
import typing


@command(module='asr.charge_neutrality',
         requires=[],
         dependencies=[],
         resources='1:10m',
         returns=ASRResult)
@option('--temp', help='Temperature [K]', type=float)
def main(temp: float = 300) -> ASRResult:
    """Calculate self-consistent Fermi energy for defect systems.

    This recipe calculates the self-consistent Fermi energy for a
    specific host system. It needs the defect folder structure
    that gets created with asr.setup.defects.
    """
    _, calc = restart('gs.gpw', txt=None)
    dos, EF, gap = get_dos(calc)
    dos = renormalize_dos(calc, dos, EF)

    n0, p0 = integrate_electron_hole_concentration(dos,
                                                   EF,
                                                   gap,
                                                   temp)

    return ASRResult()


def integrate_electron_hole_concentration(dos, ef, gap, T):
    """Integrate electron and hole carrier concentration.

    Use the trapezoid rule to integrate the following expressions:

     - n0 = int_{E_gap}^{inf} dos(E) * f_e(EF, E, T) dE
     - p0 = int_{-inf}^{0} dos(E) * f_h(EF, E, T) dE
     - f_e(EF, E, T) = (exp((EF - E) / (k * T)) + 1)^{-1}
     - f_h(EF, E, T) = 1 - f_e(EF, E, T)
    """
    # define spacing for density of states integration
    dx = abs(dos[0][-1] - dos[0][0]) / len(dos[0])

    # electron carrier concentration integration
    # emin = gap
    # emax = dos[0][-1]
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy >= gap:
            int_el.append(rho * fermi_dirac_holes(energy, ef, T))
    n0 = np.trapz(int_el, dx=dx)
    print('INFO: calculated electron carrier concentration: {}'.format(n0))

    # hole carrier concentration integration
    # emin = dos[0][0]
    # emax = 0
    int_hole = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= 0:
            int_hole.append(rho * fermi_dirac_electrons(energy, ef, T))
    p0 = np.trapz(int_hole, dx=dx)
    print('INFO: calculated hole carrier concentration: {}'.format(p0))

    return n0, p0


def renormalize_dos(calc, dos, ef):
    """Renormalize DOS according to number of electrons."""
    if calc.get_number_of_spins() == 2:
        Ne = calc.get_number_of_electrons() / 2.
    else:
        Ne = calc.get_number_of_electrons()

    Ne = calc.get_number_of_electrons()

    ### integrate number of electrons
    # emin = dos[0][0]
    # emax = 0

    dx = abs(dos[0][-1] - dos[0][0]) / len(dos[0])
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= ef:
            int_el.append(rho)
    n0 = np.trapz(int_el, dx=dx)

    print(f'INFO: number of electrons BEFORE renormalization: {n0}, '
          f'{calc.get_number_of_electrons()} (Reference).')
    for i in range(len(dos[0])):
        dos[1][i] = dos[1][i] * Ne / n0

    print('INFO: renormalize DOS with factor {}'.format(Ne / n0))

    # reintegrate up to Fermi energy to doublecheck whether number of
    # electrons is correct
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= ef:
            int_el.append(rho)
    n0 = np.trapz(int_el, dx=dx)

    print(f'INFO: number of electrons AFTER renormalization: {n0:.2f}, '
          f'{calc.get_number_of_electrons()} (Reference).')

    return dos



def get_band_edges(calc):
    """Returns energy of VBM to reference to later."""
    gap, p1, p2 = bandgap(calc)
    if gap == 0:
        raise ValueError('No bandgap for the present host material!')

    evbm = calc.get_eigenvalues(spin=p1[0], kpt=p1[1])[p1[2]]
    ecbm = calc.get_eigenvalues(spin=p2[0], kpt=p2[1])[p2[2]]
    EF = calc.get_fermi_level()

    return evbm, ecbm, EF, gap


def get_dos(calc, npts=4001, width=0.01):
    """
    Returns the density of states with energy set to zero for VBM.
    """
    dos = calc.get_dos(spin=0, npts=npts, width=width)
    if calc.get_number_of_spins() == 2:
        dos_1 = calc.get_dos(spin=1, npts=npts, width=width)
        for i in range(len(dos[0])):
            dos[1][i] = dos[1][i] + dos_1[1][i]
    else:
        print('INFO: non spin-polarized calculation! Only one spin channel present!')
    evbm, ecbm, EF, gap = get_band_edges(calc)

    # reference density of states such that E_VBM = 0
    for i in range(npts):
        dos[0][i] = dos[0][i] - evbm
    EF = EF - evbm

    return dos, EF, gap


def fermi_dirac_electrons(E, EF, T):
    _k = 8.617333262145e-5  # in eV/K

    return 1. / (np.exp((EF - E)/(_k * T)) + 1.)


def fermi_dirac_holes(E, EF, T):
    return 1 - fermi_dirac_electrons(E, EF, T)


# @prepare_result
# class Result(ASRResult):
#     """Container for charge neutrality results."""
# 
#     EF: float
#     defect_conc: typing.List[ConcResult]
# 
#     key_descriptions = dict(
#         EF='Self-consistent Fermi energy [eV].',
#         defect_conc='List of ConcResult containers.')
# 
#     formats = {"ase_webpanel": webpanel}
