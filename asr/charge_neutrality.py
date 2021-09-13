"""Self-consistent EF calculation for defect systems.."""
from asr.core import command, option, ASRResult, prepare_result, DictStr
from ase.dft.bandgap import bandgap
from asr.database.browser import make_panel_description, href
import typing
from gpaw import restart
import numpy as np


# TODO: implement test
# TODO: automate degeneracy counting


panel_description = make_panel_description(
    """
Equilibrium defect energetics evaluated by solving E<sub>F</sub> self-consistently
until charge neutrality is achieved.
""",
    articles=[
        href("""J. Buckeridge, Equilibrium point defect and charge carrier
 concentrations in a meterial determined through calculation of the self-consistent
 Fermi energy, Comp. Phys. Comm. 244 329 (2019)""",
             'https://doi.org/10.1016/j.cpc.2019.06.017'),
    ],
)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig, WebPanel,
                                      describe_entry, table,
                                      dl, code)

    table_list = []
    unit = result.conc_unit
    unitstring = f"cm<sup>{unit.split('^')[-1]}</sup>"
    for element in result.defect_concentrations:
        name = element['defect_name']
        def_type = name.split('_')[0]
        if def_type == 'v':
            def_type = 'V'
        def_name = name.split('_')[1]
        scf_table = table(result, f'Eq. concentrations of {def_type}<sub>{def_name}</sub> [{unitstring}]', [])
        for altel in element['concentrations']:
            if altel[0] > 1e1:
                scf_table['rows'].extend(
                    [[describe_entry(f'<b>Charge {altel[1]:1d}</b>',
                                     description='Equilibrium concentration '
                                                 'in charge state q at T = '
                                                 f'{int(result.temperature):d} K.'),
                      f'<b>{altel[0]:.1e}</b>']])
            else:
                scf_table['rows'].extend(
                    [[describe_entry(f'Charge {altel[1]:1d}',
                                     description='Equilibrium concentration '
                                                 'in charge state q at T = '
                                                 f'{int(result.temperature):d} K.'),
                      f'{altel[0]:.1e}']])
        table_list.append(scf_table)

    ef = result.efermi_sc
    gap = result.gap
    if ef < (gap / 4.):
        dopability = '<b style="color:red;">p-type</b>'
    elif ef > (3 * gap / 4.):
        dopability = '<b style="color:blue;">n-type</b>'
    else:
        dopability = 'intrinsic'

    # get strength of p-/n-type dopability
    if ef < 0:
        ptype_val = '100+'
        ntype_val = '0'
    elif ef > gap:
        ptype_val = '0'
        ntype_val = '100+'
    else:
        ptype_val = int((1 - ef / gap) * 100)
        ntype_val = int((100 - ptype_val))
    pn_strength = f'{ptype_val:3}% / {ntype_val:3}%'
    pn = describe_entry(
        'p-type / n-type strength',
        'Strength of p-/n-type dopability in percent '
        f'(normalized wrt. band gap) at T = {int(result.temperature):d} K.'
        + dl(
            [
                [
                    '100/0',
                    code('if E<sub>F</sub> at VBM')
                ],
                [
                    '0/100',
                    code('if E<sub>F</sub> at CBM')
                ],
                [
                    '50/50',
                    code('if E<sub>F</sub> at E<sub>gap</sub> * 0.5')
                ]
            ],
        )
    )

    is_dopable = describe_entry(
        'Intrinsic doping type',
        'Is the material intrinsically n-type, p-type or intrinsic at '
        f'T = {int(result.temperature):d} K?'
        + dl(
            [
                [
                    'p-type',
                    code('if E<sub>F</sub> < 0.25 * E<sub>gap</sub>')
                ],
                [
                    'n-type',
                    code('if E<sub>F</sub> 0.75 * E<sub>gap</sub>')
                ],
                [
                    'intrinsic',
                    code('if 0.25 * E<sub>gap</sub> < E<sub>F</sub> < '
                         '0.75 * E<sub>gap</sub>')
                ],
            ],
        )
    )

    scf_fermi = describe_entry(
        'Fermi level position',
        'Self-consistent Fermi level wrt. VBM at which charge neutrality condition is '
        f'fulfilled at T = {int(result.temperature):d} K [eV].')

    scf_summary = table(result, 'Charge neutrality', [])
    scf_summary['rows'].extend([[is_dopable, dopability]])
    scf_summary['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_summary['rows'].extend([[pn, pn_strength]])

    scf_overview = table(result,
                         f'Equilibrium properties @ {int(result.temperature):d} K', [])
    scf_overview['rows'].extend([[is_dopable, dopability]])
    scf_overview['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_overview['rows'].extend([[pn, pn_strength]])
    if result.n0 > 1e-5:
        n0 = result.n0
    else:
        n0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Electron carrier concentration',
                         'Equilibrium electron carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{n0:.1e} {unitstring}']])
    if result.p0 > 1e-5:
        p0 = result.p0
    else:
        p0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Hole carrier concentration',
                         'Equilibrium hole carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{p0:.1e} {unitstring}']])

    figure = describe_entry(fig('charge_neutrality.png'),
                            description='Formation energies of all point defects '
                                        'intrinsic to the material and self-consistent '
                                        'Fermi level at T = '
                                        f'{int(result.temperature):d} K.')

    panel = WebPanel(
        describe_entry('Equilibrium defect energetics',
                       panel_description),
        columns=[[figure, scf_overview], table_list],
        plot_descriptions=[{'function': plot_formation_scf,
                            'filenames': ['charge_neutrality.png']}],
        sort=25)

    summary = {'title': 'Summary',
               'columns': [[scf_summary], []],
               'sort': 1}

    return [panel, summary]


@prepare_result
class ConcentrationResult(ASRResult):
    """Container for concentration results of a specific defect."""

    defect_name: str
    concentrations: typing.List[typing.Tuple[float, float, int]]

    key_descriptions = dict(
        defect_name='Name of the defect ({position}_{type}).',
        concentrations='List of concentration tuples containing (conc., eform, '
                       'chargestate).')


@prepare_result
class Result(ASRResult):
    """Container for asr.charge_neutrality results."""

    temperature: float
    efermi_sc: float
    gap: float
    n0: float
    p0: float
    defect_concentrations: typing.List[ConcentrationResult]
    conc_unit: str
    dopability: str

    key_descriptions: typing.Dict[str, str] = dict(
        temperature='Temperature [K].',
        efermi_sc='Self-consistent Fermi level at which charge '
                  'neutrality condition is fulfilled [eV].',
        gap='Electronic band gap [eV].',
        n0='Electron carrier concentration at SC Fermi level.',
        p0='Hole carrier concentration at SC Fermi level.',
        defect_concentrations='List of ConcentrationResult containers.',
        conc_unit='Unit of calculated concentrations.',
        dopability='p-/n-type or intrinsic nature of material.')

    formats = {"ase_webpanel": webpanel}


@command(module='asr.charge_neutrality',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         resources='1:10m',
         returns=ASRResult)
@option('--temp', help='Temperature [K]', type=float)
@option('--defects', help='Defect dictionary.', type=DictStr())
def main(temp: float = 300,
         defects: dict = {}) -> ASRResult:
    """Calculate self-consistent Fermi energy for defect systems.

    This recipe calculates the self-consistent Fermi energy for a
    specific host system. It needs the defect folder structure
    that gets created with asr.setup.defects. If you do not have a
    defect folder structure present, please use the '--defects' option.
    It is structured the following way (at the example of MoS2):

    - defect_dict = {'defect_name': [(form. en. at VBM, charge state), (.., ..)],
                     ...}
    """
    # test input and read in defect dictionary from asr.sj_analyze results
    if defects == {}:
        defectdict = return_defect_dict()
    else:
        defectdict = defects

    # read in pristine ground state calculation and evaluate,
    # renormalize density of states
    atoms, calc = restart('gs.gpw', txt=None)
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

    # evaluate units based on the dimensionality of the system
    dim = np.sum(atoms.get_pbc())
    if dim == 2:
        unit = 'cm^-2'
    elif dim == 3:
        unit = 'cm^-3'

    # if calculation is converged, show final results
    if converged:
        n0, p0 = integrate_electron_hole_concentration(dos,
                                                       E,
                                                       gap,
                                                       temp)
        n0 = convert_concentration_units(n0, atoms)
        p0 = convert_concentration_units(p0, atoms)
        print(f'INFO: Calculation converged after {i} steps! Final results:')
        print(f'      Self-consistent Fermi-energy: {E:.2f} eV.')
        print(f'      Equilibrium electron concentration: {n0:.2e} {unit:5}.')
        print(f'      Equilibrium hole concentration: {p0:.2e} {unit:5}.')
        print(f'      Concentrations:')
        concentration_results = []
        for defecttype in defectdict:
            print(f'      - defecttype: {defecttype}')
            print(f'      ----------------------------------------------')
            concentration_tuples = []
            for defect in defectdict[defecttype]:
                eform = get_formation_energy(defect, E)
                conc_def = calculate_defect_concentration(eform,
                                                          1,
                                                          1,
                                                          1,
                                                          temp)
                conc_def = convert_concentration_units(conc_def, atoms)
                concentration_tuples.append((conc_def, int(defect[1]), eform))
                print(f'      defect concentration for ({defect[1]:2}): {conc_def:.2e} {unit:5}')
            concentration_result = ConcentrationResult.fromdata(
                defect_name=defecttype,
                concentrations=concentration_tuples)
            concentration_results.append(concentration_result)

    if E < 0.25 * gap:
        dop = 'p-type'
    elif E > 0.75 * gap:
        dop = 'n-type'
    else:
        dop = 'intrinsic'

    return Result.fromdata(
        temperature=temp,
        efermi_sc=E,
        gap=gap,
        n0=n0,
        p0=p0,
        defect_concentrations=concentration_results,
        conc_unit=unit,
        dopability=dop)


def convert_concentration_units(conc, atoms):
    """
    Convert concentration to units on cm^-n.

    Note, that n is the dimensionality of the system.
    """
    # cell = atoms.get_cell()
    volume = atoms.get_volume()
    dim = np.sum(atoms.get_pbc())

    # conversion factor from \AA to cm
    ang_to_cm = 1. * 10 ** (-8)

    if dim == 1:
        raise NotImplementedError('Not implemented for 1D structures!')
    elif dim == 2:
        z = atoms.get_cell_lengths_and_angles()[2]
        volume = volume / z
        conc = conc / (volume * (ang_to_cm ** 2))
    elif dim == 3:
        volume = volume
        conc = conc / (volume * (ang_to_cm ** 3))

    return conc


def fermi_dirac_electrons(E, EF, T):
    _k = 8.617333262145e-5  # in eV/K

    return 1. / (np.exp((EF - E) / (_k * T)) + 1.)


def fermi_dirac_holes(E, EF, T):
    return 1 - fermi_dirac_electrons(E, EF, T)


def calculate_delta(conc_list, chargelist, n0, p0):
    """
    Calculate charge balance for current energy.

    delta = n_0 - p_0 - sum_X(sum_q C_{X^q}).
    """
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
    """Calculate and return the defect concentration for a specific defect.

    For a particular charge state with the formation energy for a particular energy.

    Use C_X^q = N_X * g_{X^q} * exp(-E_f(X^q) / (k * T))

    Note, that e_form is the formation energy at the desired Fermi energy
    already! In general, it is not the one at the VBM.
    """
    _k = 8.617333262145e-5  # in eV/K
    return (sites * degeneracy * np.exp((-1) * e_form / (_k * T)))


def get_formation_energy(defect, energy):
    """Return formation energy of defect in a charge state at an energy."""
    E_form_0, charge = get_zero_formation_energy(defect)

    return E_form_0 + charge * energy


def get_zero_formation_energy(defect):
    """
    Return the formation energy of a given defect at the VBM.

    Note, that the VBM corresponds to energy zero.
    """
    eform = defect[0]
    charge = defect[1]

    return eform, charge


def return_defect_dict():
    """Read in the results of asr.sj_analyze and store the formation energies at VBM."""
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
    """
    Integrate electron and hole carrier concentration.

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
    """Return energy of VBM to reference to later."""
    gap, p1, p2 = bandgap(calc)
    if gap == 0:
        raise ValueError('No bandgap for the present host material!')

    evbm = calc.get_eigenvalues(spin=p1[0], kpt=p1[1])[p1[2]]
    ecbm = calc.get_eigenvalues(spin=p2[0], kpt=p2[1])[p2[2]]
    EF = calc.get_fermi_level()

    return evbm, ecbm, EF, gap


def get_dos(calc, npts=4001, width=0.01):
    """Return the density of states with energy set to zero for VBM."""
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


def plot_formation_scf(row, fname):
    """Plot formation energy diagram and SC Fermi level wrt. VBM."""
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.charge_neutrality.json')

    ef = data['efermi_sc']
    gap = data['gap']
    for i, defect in enumerate(data['defect_concentrations']):
        name = defect['defect_name']
        def_type = name.split('_')[0]
        def_name = name.split('_')[-1]
        if def_type == 'v':
            def_type = 'V'
        namestring = f"{def_type}$_\\{'mathrm{'}{def_name}{'}'}$"
        plt.plot([], [], linestyle='solid', color=f'C{i}', label=namestring)
        for conc_tuple in defect['concentrations']:
            q = conc_tuple[1]
            eform = conc_tuple[2]
            y0 = q * (-ef) + eform
            y1 = q * (gap - ef) + eform
            plt.plot([0, gap], [y0, y1], linestyle='solid', color=f'C{i}')

    plt.axvline(0, color='black')
    plt.axvline(gap, color='black')
    plt.axvspan(-100, 0, alpha=0.5, color='grey')
    plt.axvspan(gap, 100, alpha=0.5, color='grey')
    plt.axvline(ef, color='red', linestyle='dotted', label=r'$E_\mathrm{F}^{\mathrm{sc}}$')
    plt.xlim(0 - gap / 10., gap + gap / 10.)
    yminold = plt.gca().get_ylim()[0]
    plt.ylim(yminold, yminold + 4)
    plt.xlabel(r'$E_\mathrm{F} - E_{\mathrm{VBM}}$ [eV]')
    plt.ylabel(r'$E^f$ [eV] (wrt. standard states)')
    # plt.legend(ncol=2, loc=9)
    plt.legend(bbox_to_anchor=(0.5, 1.1), ncol=5, loc='lower center')
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
