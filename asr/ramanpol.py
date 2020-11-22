
import numpy as np
import sys
import typing
from pathlib import Path

from ase.io import read
from ase.parallel import parprint
from asr.core import (command, option, DictStr, ASRResult,
                      read_json, write_json, prepare_result)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import fig

    # Make a table from the phonon modes
    data = row.data.get('results-asr.ramanpol.json')
    if data:
        table = []
        freqs_l = data['freqs_l']
        w_l, rep_l = count_deg(freqs_l)
        nph = len(w_l)
        for ii in range(nph):
            key = 'Mode {}'.format(ii + 1)
            table.append(
                (key,
                 np.array2string(
                     np.abs(w_l[ii]), precision=1), rep_l[ii]))
        opt = {'type': 'table',
               'header': ['Mode', 'Frequency (1/cm)', 'Degeneracy'],
               'rows': table}
    else:
        opt = None
    # Make the panel
    panel = {'title': 'Raman spectrum (RPA)',
             'columns': [[fig('Raman.png')], [opt]],
             'plot_descriptions':
                 [{'function': plot_raman, 'filenames': ['Raman.png']}],
             'sort': 22}

    return [panel]


@prepare_result
class Result(ASRResult):

    freqs_l: typing.List[float]
    wavelength_w: typing.List[float]
    amplitudes_vvwl: typing.List[typing.List[typing.List[typing.List[complex]]]]
    irrep_l: typing.List

    key_descriptions = {
        'freqs_l': 'Phonon frequencies (the Gamma point) [1/cm]',
        'wavelength_w': 'Laser excitation wavelength [nm]',
        'amplitudes_vvwl': 'Raman tensor [a.u.]',
        'irrep_l': 'Irreducible representation label',
    }
    formats = {'ase_webpanel': webpanel}


@command(
    'asr.ramanpol',
    requires=['structure.json'],
    returns=Result)
@option('--disp', type=float, help='Displacement size')
@option('--dftd3', type=bool, help='Enable DFT-D3 for phonon calculations')
@option('--wavelengths', type=list, help='Excitation wavelengths [nm]')
@option('--eta', type=float, help='Excitation broadening [eV]')
@option('--prefix', type=str, help='Prefix for filenames')
@option('--calc_ph', type=DictStr(), help='Calculator params. for phonon')
@option('--calc_chi', type=DictStr(), help='Calculator params. for chi')
@option('--removefiles', type=str, help='Remove intermediate files')
def main(
    disp: float = 0.05, eta: float = 0.1, dftd3: bool = True,
    wavelengths: typing.List[float] = [
        488.0, 532.0, 594.0, 612.0, 633.0, 708.0, 780.0,
        850.0, 1064.0, 1550.0, 2600.0, 4800.0, 10600.0],
    removefiles: str = 'gs', prefix: str = '',
    calc_ph: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 6.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac', 'width': 0.05},
        'convergence': {'forces': 1.0e-5},
        'symmetry': {'point_group': False},
        'txt': 'gs_phonons.txt'},
    calc_chi: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac', 'width': 0.05},
        'nbands': '200%',
        'convergence': {'bands': -5},
        'symmetry': {'point_group': False},
        'txt': 'gs_chi.txt'}, ) -> ASRResult:

    from ase.utils.timing import Timer
    from gpaw.mpi import world
    try:
        timer = Timer()
        with timer('Phonon calculations'):
            parprint('Starting a phonon calculation ...')
            if not Path(prefix + 'phonons.json').is_file():
                find_phonons(calculator=calc_ph, dftd3=dftd3,
                             disp=disp, prefix=prefix)
            ph = read_json(prefix + 'phonons.json')
            freqs_l = ph['freqs_l']
            u_lav = ph['u_lav']
            irrep_l = ph['irrep_l']
            parprint('The phonons are calculated and loaded.')
            if world.rank == 0:
                if removefiles == 'all':
                    for ff in Path().glob(prefix + 'phonons.*.json'):
                        if ff.is_file():
                            ff.unlink()

        with timer('Raman calculations'):
            parprint('Starting a Raman calculation ...')
            freqs_w = [1240 / wavelength for wavelength in wavelengths]
            if not Path(prefix + 'chis.json').is_file():
                I_vvwl = get_raman_tensor(
                    calc_chi, freqs_l, u_lav, prefix=prefix, eta=eta,
                    removefiles=removefiles, freqs_w=freqs_w, disp=disp)
                parprint('The Raman tensors are fully computed.')
            chis = read_json(prefix + 'chis.json')
            I_vvwl = chis['I_vvwl']

        # Make the output data
        results = {
            'amplitudes_vvwl': I_vvwl,
            'wavelength_w': wavelengths,
            'freqs_l': freqs_l,
            'irrep_l': irrep_l, }

    finally:
        if world.rank == 0:
            timer.write()

    return results


def calcspectrum(wavelength, w_l, I_l, ww, gamma=3, shift=0, temp=300):
    from ase.units import kB

    # Gaussian function definition
    def gauss(w, g):
        gauss = 1 / (g * np.sqrt(2 * np.pi)) * np.exp(-0.5 * w**2 / g**2)
        gauss[gauss < 1e-16] = 0
        return gauss

    cm = 1 / 8065.544
    kbT = kB * temp / cm
    rr = np.zeros(np.size(ww))
    freq = 1240 / wavelength / cm
    for wi, ri in zip(w_l, I_l):
        if wi > 1e-1:
            nw = 1 / (np.exp(wi / kbT) - 1)
            curr = (1 + nw) * np.abs(ri)**2 / wi * (freq - wi)**4
            rr = rr + curr * gauss(ww - wi - shift, gamma)
    return rr


def plot_raman(row, filename):
    # Import the required modules
    import matplotlib.pyplot as plt

    # All required settings
    params = {'broadening': 1.0,  # in cm^-1
              'wavelength': 2,  # index
              'polarization': ['xx', 'yy', 'zz'],
              'temperature': 300}  # in K

    # Read the data from the disk
    data = row.data.get('results-asr.ramanpol.json')

    # Set the variables and parameters
    wavelength_w = data['wavelength_w']
    freqs_l = data['freqs_l']
    amplitudes_vvwl = data['amplitudes_vvwl']
    irrep_l = data['irrep_l']
    selpol = params['polarization']
    gamma = params['broadening']
    waveind = params['wavelength']

    # Check the data to be consistent
    ampshape = amplitudes_vvwl.shape
    freqshape = len(freqs_l)
    waveshape = len(wavelength_w)
    if (ampshape[0] != 3) or (ampshape[1] != 3) or (
            ampshape[2] != waveshape) or (ampshape[3] != freqshape):
        return

    # Make the spectrum
    maxw = min([int(np.max(freqs_l) + 200), int(1.2 * np.max(freqs_l))])
    minw = -maxw / 100
    ww = np.linspace(minw, maxw, 2 * maxw)
    rr = {}
    maxr = np.zeros(len(selpol))
    for ii, pol in enumerate(selpol):
        d_i = 'xyz'.index(pol[0])
        d_o = 'xyz'.index(pol[1])
        rr[pol] = calcspectrum(
            wavelength_w[waveind], freqs_l, amplitudes_vvwl[d_i, d_o, waveind],
            ww, gamma=gamma, temp=params['temperature'])
        maxr[ii] = np.max(rr[pol])

    # Make the figure panel and add y=0 axis
    ax = plt.figure().add_subplot(111)
    ax.axhline(y=0, color="k")

    # Plot the data and add the axis labels
    for ipol, pol in enumerate(selpol):
        if ipol > 2:
            sty = '--'
        else:
            sty = '-'
        ax.plot(ww, rr[pol] / np.max(maxr), sty, c='C' + str(ipol), label=pol)
    ax.set_xlabel('Raman shift [cm$^{-1}$]')
    ax.set_ylabel('Raman intensity [a.u.]')
    ax.set_ylim((-0.1, 1.1))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim((minw, maxw))

    # Add the legend to figure
    ax.legend()

    # Count the modes and their degeneracy factors
    w_l, rep_l = count_deg(freqs_l)

    # Add the phonon bars to the figure with showing their degeneracy factors
    for irrep, freq in zip(irrep_l, freqs_l):
        if irrep:
            irrep2 = ''
            for ch in irrep:
                if ch.isdigit() or ch == 'g' or ch == 'u':
                    irrep2 += '$_' + ch + '$'
                else:
                    irrep2 += ch
            plt.text(freq, -0.1, irrep2, ha='center', va='bottom', rotation=0)
    plt.bar(w_l, -0.04, width=maxw / 100, color='k')
    # for idx, rect in enumerate(pltbar):
    #     ax.text(rect.get_x() + rect.get_width() / 2., -0.1,
    #             str(int(rep_l[idx])), ha='center', va='bottom', rotation=0)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename)


def count_deg(freqs_l, freq_err=2):

    # Degeneracy factor for modes
    w_l = [freqs_l[0]]
    rep_l = [1]
    # Loop over modes
    for wss in freqs_l[1:]:
        ind = len(w_l) - 1
        if np.abs(w_l[ind] - wss) > freq_err:
            w_l.append(wss)
            rep_l.append(1)
        else:
            rep_l[ind] += 1
    w_l = np.array(w_l)
    rep_l = np.array(rep_l)
    # Return the output
    return w_l, rep_l


def symmetrize_chi(atoms, chi_vvl):
    import spglib
    sg = spglib.get_symmetry((
        atoms.cell,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers()), symprec=1e-4)
    op_scc = sg['rotations']
    cell_cv = atoms.cell
    op_svv = [np.linalg.inv(cell_cv).dot(op_cc.T).dot(cell_cv) for
              op_cc in op_scc]
    nop = len(op_svv)
    sym_chi_vvl = np.zeros_like(chi_vvl)
    for op_vv in op_svv:
        sym_chi_vvl += np.einsum('il,jm,lmn->ijn',
                                 op_vv, op_vv, chi_vvl)

    return sym_chi_vvl / nop


def get_raman_tensor(
    calculator, freqs_l, u_lav, removefiles='no', prefix='',
        freqs_w=[0.0, 2.33], eta=0.05, disp=0.05):

    from ase.calculators.calculator import get_calculator_class
    from gpaw.nlopt.matrixel import make_nlodata
    from gpaw.mpi import world

    name = calculator.pop('name')
    calculator['txt'] = prefix + calculator['txt']
    calc = get_calculator_class(name)(**calculator)

    atoms = read('structure.json')
    mass_a = atoms.get_masses()
    I_vvwl = np.zeros((3, 3, len(freqs_w), len(freqs_l)), complex)
    set_chi_ivvw = []
    for mode, freq in enumerate(freqs_l):
        if freq < 0.1:
            continue
        chi_ivvw = np.zeros((2, 3, 3, len(freqs_w)), complex)

        # Make the displaced structure
        atoms_new = atoms.copy()
        pos_Nav = atoms_new.get_positions()
        mode_av = np.real(u_lav[mode])
        n_a = np.linalg.norm(mode_av, axis=1)
        mode_av /= np.max(n_a)
        mode_av /= np.sqrt(mass_a)[:, np.newaxis]

        for ind in range(2):
            atoms_new.set_positions(pos_Nav + mode_av * disp * (2 * ind - 1))
            sign = ['+', '-'][ind % 2]
            mml_name = prefix + f'mml_{mode}{sign}.npz'
            if not Path(mml_name).is_file():
                gs_name = prefix + f'gs_{mode}{sign}.gpw'
                if not Path(gs_name).is_file():
                    parprint(f'Starting GS calculation for M{mode}{sign} ...')
                    atoms_new.calc = calc
                    atoms_new.get_potential_energy()
                    calc.write(gs_name, 'all')
                    parprint(f'The GS calculation for M{mode}{sign} is done.')
                    sys.stdout.flush()

                # Calculate momentum matrix:
                make_nlodata(gs_name=gs_name, out_name=mml_name)
                if world.rank == 0:
                    if removefiles == 'all' or removefiles == 'gs':
                        ff = Path(gs_name)
                        if ff.is_file():
                            ff.unlink()

            parprint(f'Starting a chi calculation for mode {mode}{sign} ...')
            chi_vvw = get_chi_tensor(
                freqs=freqs_w, eta=eta,
                mml_name=mml_name)
            sym_chi_vvw = symmetrize_chi(atoms_new, chi_vvw)
            # sym_chi_vvw = chi_vvw
            parprint(f'The chi calculation for mode {mode}{sign} is done.')
            sys.stdout.flush()
            if removefiles == 'all' or removefiles == 'chi':
                ff = Path(mml_name)
                if ff.is_file():
                    ff.unlink()

            chi_ivvw[ind] = sym_chi_vvw

        set_chi_ivvw.append(chi_ivvw)
        alp_vvw = np.abs(chi_ivvw[1] - chi_ivvw[0]) / (2 * disp)
        I_vvwl[:, :, :, mode] = alp_vvw

    results = {
        'I_vvwl': I_vvwl,
        'freqs_w': freqs_w,
        'chi_livvw': np.array(set_chi_ivvw),
        'disp': disp}
    write_json(prefix + 'chis.json', results)

    return I_vvwl


def find_phonons(calculator, dftd3=False, disp=0.05, prefix=''):

    from ase.calculators.calculator import get_calculator_class
    from ase.calculators.dftd3 import DFTD3
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.units import THzToCm
    from ase.atoms import Atoms
    from gpaw.mpi import world

    name = calculator.pop('name')
    calculator['txt'] = prefix + calculator['txt']
    calc = get_calculator_class(name)(**calculator)
    if dftd3:
        calc = DFTD3(dft=calc, cutoff=60)

    # Remove empty files:
    if world.rank == 0:
        for ff in Path().glob(prefix + 'phonons.*.json'):
            if ff.stat().st_size == 0:
                ff.unlink()

    # Make the Phonopy object
    atoms = read('structure.json')
    supercell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.symbols,
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions())

    phonon = Phonopy(phonopy_atoms, supercell)
    phonon.generate_displacements(distance=disp, is_plusminus=True)
    displaced_sc = phonon.get_supercells_with_displacements()
    scell = displaced_sc[0]
    atoms_new = Atoms(
        symbols=scell.get_chemical_symbols(),
        scaled_positions=scell.get_scaled_positions(),
        cell=scell.get_cell(),
        pbc=atoms.pbc)

    set_of_forces = []
    for ind, cell in enumerate(displaced_sc):
        # Sign of the displacement
        sign = ['+', '-'][ind % 2]
        filename = prefix + 'phonons.{0}{1}.json'.format(ind // 2, sign)
        if Path(filename).is_file():
            forces = read_json(filename)['force']
        else:
            atoms_new.set_scaled_positions(cell.get_scaled_positions())
            atoms_new.calc = calc
            forces = atoms_new.get_forces()

        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        write_json(filename, {'force': forces})
        set_of_forces.append(forces)

    phonon.produce_force_constants(
        forces=set_of_forces, calculate_full_force_constants=False)
    phonon.symmetrize_force_constants()

    # Get the phonons
    q_c = np.array([0, 0, 0])
    freqs_l, u_ll = phonon.get_frequencies_with_eigenvectors(q_c)
    freqs_l *= THzToCm
    u_lav = u_ll.T.reshape(3 * len(atoms), len(atoms), 3)
    phonon.set_irreps(q_c)
    ob = phonon._irreps
    irreps = []
    for _, (deg, irr) in enumerate(
            zip(ob._degenerate_sets, ob._ir_labels)):
        irreps += [irr] * len(deg)

    results = {
        'freqs_l': freqs_l,
        'u_lav': u_lav,
        'irrep_l': irreps}
    write_json(prefix + 'phonons.json', results)


def get_chi_tensor(
        freqs=[1.0], eta=0.05,
        ftol=1e-4, Etol=1e-6,
        band_n=None, mml_name='mml.npz'):

    from gpaw.nlopt.basic import load_data
    from gpaw.mpi import world

    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = freqs + 1e-12 + 1j * eta  # Add small value to avoid 0

    # Load the required data
    k_info = load_data(mml_name=mml_name)
    _, tmp = k_info.popitem()
    nb = len(tmp[1])
    if band_n is None:
        band_n = list(range(nb))

    # Initialize the outputs
    sum_vvl = np.zeros((3, 3, nw), complex)

    # Do the calculations
    for _, (we, f_n, E_n, p_vnn) in k_info.items():
        tmp = np.zeros((3, 3, nw), complex)
        for v1 in range(3):
            for v2 in range(v1, 3):
                sum_l = calc_chi(
                    w_lc, f_n, E_n, p_vnn, [v1, v2],
                    band_n, ftol, Etol, eshift=0)
                tmp[v1, v2] = sum_l
                tmp[v2, v1] = sum_l
        # Add it to previous with a weight
        sum_vvl += tmp * we

    world.sum(sum_vvl)
    chi_vvl = sum_vvl

    return chi_vvl


def calc_chi(
    w_l, f_n, E_n, p_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    sum_l = np.zeros(w_l.size, complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift
            if np.abs(fnm) < ftol or np.abs(Emn) < Etol:
                continue
            # *2 for real, /2 for TRS
            sum_l += 2 * fnm * np.real(
                p_vnn[pol_v[0], nni, mmi] * p_vnn[pol_v[1], mmi, nni]) \
                / (Emn * (w_l**2 - Emn**2))

    return sum_l


if __name__ == "__main__":
    main.cli()
