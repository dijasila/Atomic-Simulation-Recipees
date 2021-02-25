"""Self-consistent EF calculation for defect systems.."""
from asr.core import command, option, ASRResult, prepare_result
from ase.dft.bandgap import bandgap
import typing
from gpaw import restart
import numpy as np


# TODO: add plotting routines for formation energies and SC EF
# TODO: implement Results
# TODO: implement Webpanel
# TODO: implement test
# TODO: automate degeneracy counting


@prepare_result
class ConcentrationResult(ASRResult):
    """Container for concentration results of a specific defect."""
    defect_name: str
    concentrations: typing.List[typing.Tuple[float, int]]

    key_descriptions = dict(
        defect_name='Name of the defect ({position}_{type}).',
        concentrations='List of concentration tuples containing (conc., chargestate).')


@prepare_result
class Result(ASRResult):
    """Container for asr.charge_neutrality results."""
    temperature: float
    efermi_sc: float
    n0: float
    p0: float
    defect_concentrations: typing.List[ConcentrationResult]

    key_descriptions: typing.Dict[str, str] = dict(
        temperature='Temperature [K].',
        efermi_sc='Self-consistent Fermi level at which charge '
                  'neutrality condition is fulfilled [eV].',
        n0='Electron carrier concentration at SC Fermi level [eV].',
        p0='Hole carrier concentration at SC Fermi level [eV].',
        defect_concentrations='List of ConcentrationResult containers.')


@command(module='asr.charge_neutrality',
         requires=['gs.gpw'],
         dependencies=['asr.gs'],
         resources='1:10m',
         returns=ASRResult)
@option('--temp', help='Temperature [K]', type=float)
def main(temp: float = 300) -> ASRResult:
    """Calculate self-consistent Fermi energy for defect systems.

    This recipe calculates the self-consistent Fermi energy for a
    specific host system. It needs the defect folder structure
    that gets created with asr.setup.defects.
    """
    # test input and read in defect dictionary from asr.sj_analyze results
    defectdict = return_defect_dict()

    # read in pristine ground state calculation and evaluate,
    # renormalize density of states
    _, calc = restart('gs.gpw', txt=None)
    dos, EF, gap = get_dos(calc)
    dos = renormalize_dos(calc, dos, EF)

    # Calculate initial electron and hole carrier concentration
    n0, p0 = integrate_electron_hole_concentration(dos,
                                                   EF,
                                                   gap,
                                                   temp)

    # Initialize self-consistent loop for finding Fermi energy
    E = 0
    d = 1  # directional parameter
    i = 0  # loop index
    maxsteps = 1000  # maximum number of steps for SCF loop
    E_step = gap / 10.  # initial step sizew
    epsilon = 1e-12  # threshold for minimum step length
    converged = False  # boolean to see whether calculation is converged

    # Start the self-consistent loop
    while (i < maxsteps):
        E = E + E_step * d
        n0, p0 = integrate_electron_hole_concentration(dos,
                                                       E,
                                                       gap,
                                                       temp)
        # initialise lists for concentrations and charges
        conc_list = []
        charge_list = []
        # loop over all defects
        # FIX #
        sites = 1
        degeneracy = 1
        # FIX #
        for defecttype in defectdict:
            for defect in defectdict[defecttype]:
                eform = get_formation_energy(defect, E)
                conc_def = calculate_defect_concentration(eform,
                                                          defect[1],
                                                          sites,
                                                          degeneracy,
                                                          temp)
                conc_list.append(conc_def)
                charge_list.append(defect[1])
        # calculate delta
        delta_new = calculate_delta(conc_list, charge_list, n0, p0)
        if check_delta_zero(delta_new, conc_list, n0, p0):
            print('INFO: charge balance approximately zero! Solution found!')
            converged = True
            break
        if E_step < epsilon:
            print(f'INFO: steps smaller than threshold! Solution found!')
            converged = True
            break
        if i == 0:
            delta_old = delta_new
        elif i > 0:
            if abs(delta_new) > abs(delta_old):
                E_step = E_step / 10.
                d = -1 * d
            delta_old = delta_new
        i += 1

    # if calculation is converged, show final results
    if converged:
        n0, p0 = integrate_electron_hole_concentration(dos,
                                                       E,
                                                       gap,
                                                       temp)
        print(f'INFO: Calculation converged after {i} steps! Final results:')
        print(f'      Self-consistent Fermi-energy: {E:.4f} eV.')
        print(f'      Equilibrium electron concentration: {n0:.4e}.')
        print(f'      Equilibrium hole concentration: {p0:.4e}.')
        print(f'      Concentrations:')
        concentration_results = []
        for defecttype in defectdict:
            print(f'      - defecttype: {defecttype}')
            print(f'      --------------------------------')
            concentration_tuples = []
            for defect in defectdict[defecttype]:
                eform = get_formation_energy(defect, E)
                conc_def = calculate_defect_concentration(eform,
                                                          1,
                                                          1,
                                                          1,
                                                          temp)
                concentration_tuples.append((conc_def, int(defect[1])))
                print(f'      defect concentration for ({defect[1]:2}): {conc_def:.4e}')
            concentration_result = ConcentrationResult.fromdata(
                defect_name=defecttype,
                concentrations=concentration_tuples)
            concentration_results.append(concentration_result)

    return Result.fromdata(
        temperature=temp,
        efermi_sc=E,
        n0=n0,
        p0=p0,
        defect_concentrations=concentration_results)


def fermi_dirac_electrons(E, EF, T):
    _k = 8.617333262145e-5  # in eV/K

    return 1. / (np.exp((EF - E) / (_k * T)) + 1.)


def fermi_dirac_holes(E, EF, T):
    return 1 - fermi_dirac_electrons(E, EF, T)


def calculate_delta(conc_list, chargelist, n0, p0):
    """Calculate charge balance for current energy.

    delta = n_0 - p_0 - sum_X(sum_q C_{X^q})."""

    delta = n0 - p0
    for i, c in enumerate(conc_list):
        delta = delta - c * chargelist[i]

    return delta


def check_delta_zero(delta_new, conc_list, n0, p0):
    argument = n0 + p0
    for c in conc_list:
        argument = argument - c
    if abs(delta_new) < abs(argument * 1e-12):
        return True
    else:
        return False


def calculate_defect_concentration(e_form, charge, sites, degeneracy, T):
    """Calculates and returns the defect concentration for a specific defect in
    a particular charge state with the formation energy for a particular energy.

    Use C_X^q = N_X * g_{X^q} * exp(-E_f(X^q) / (k * T))

    Note, that e_form is the formation energy at the desired Fermi energy
    already! In general, it is not the one at the VBM.
    """
    _k = 8.617333262145e-5  # in eV/K
    return (sites * degeneracy * np.exp((-1) * e_form / (_k * T)))


def get_formation_energy(defect, energy):
    """Returns the formation energy of a given defect in a charge state at
    an specific energy."""
    E_form_0, charge = get_zero_formation_energy(defect)

    return E_form_0 + charge * energy


def get_zero_formation_energy(defect):
    """Returns the formation energy of a given defect at the VBM.

    Note, that the VBM corresponds to energy zero."""
    eform = defect[0]
    charge = defect[1]

    return eform, charge


def return_defect_dict():
    """Function that reads in the results of asr.sj_analyze and stores the formation
    energies at the VBM, together with the respective charge state."""
    from asr.core import read_json
    from pathlib import Path

    p = Path('.')
    charged_folders = list(p.glob('./../defects.*/charge_0/'))
    sjflag = False

    defect_dict = {}
    for folder in charged_folders:
        respath = Path(folder / 'results-asr.sj_analyze.json')
        if respath.is_file():
            res = read_json(respath)
            defect_name = str(folder.absolute()).split('/')[-2].split('.')[-1]
            defect_dict[defect_name] = res['eform']
            sjflag = True

    if not sjflag:
        raise RuntimeError('No SJ results available for this material! Did you run '
                           'all preliminary calculation for this system?')

    print(f'INFO: read in formation energies of the defects: {defect_dict}.')

    return defect_dict


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
    int_el = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy >= gap:
            int_el.append(rho * fermi_dirac_holes(energy, ef, T))
    n0 = np.trapz(int_el, dx=dx)
    # print('INFO: calculated electron carrier concentration: {}'.format(n0))

    # hole carrier concentration integration
    int_hole = []
    for i in range(len(dos[0])):
        energy = dos[0][i]
        rho = dos[1][i]
        if energy <= 0:
            int_hole.append(rho * fermi_dirac_electrons(energy, ef, T))
    p0 = np.trapz(int_hole, dx=dx)
    # print('INFO: calculated hole carrier concentration: {}'.format(p0))

    return n0, p0


def renormalize_dos(calc, dos, ef):
    """Renormalize DOS according to number of electrons."""
    if calc.get_number_of_spins() == 2:
        Ne = calc.get_number_of_electrons() / 2.
    else:
        Ne = calc.get_number_of_electrons()

    Ne = calc.get_number_of_electrons()

    # integrate number of electrons
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
