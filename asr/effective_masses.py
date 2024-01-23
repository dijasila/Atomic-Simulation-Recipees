from ase.io import read as ase_read
import numpy as np
import time
from ase.parallel import parprint, paropen, world
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize as pltnorm
import os
import json
from pathlib import Path
from asr.core import command, ASRResult, prepare_result, option
from asr.database.browser import fig, make_panel_description, describe_entry, \
    href
from scipy.constants import eV, m_e, hbar
from asr.magnetic_anisotropy import get_spin_axis

Mecholsky2016 = href('Mecholsky et al. (2016)',
                     'https://www.nature.com/articles/srep22098')
Mecholsky2014 = href(
    'Mecholsky et al. (2014)',
    'https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.155131')
hbar2pi = hbar * 2 * np.pi
unit_to_electron_mass = hbar2pi**2 / (2 * m_e) * 1e20 / eV

panel_description = make_panel_description(
    """
     The effective mass tensor represents the second derivative of the band
      energy w.r.t. wave vector at a band extremum. The effective masses of the
     valence bands (VB) and conduction bands (CB) are obtained as the
     eigenvalues of the mass tensor. The latter is determined by fitting a
      2nd order polynomium to the band energies on a fine k-point mesh around
      the band extrema. Spin–orbit interactions are included. The degree to
      which the band may be described by a second-order polynomial is
      quantified by the warping parameter. If the dimensionless warping
      parameter is greater than 0.0015, the effective mass is instead
      quantified by fitting to spherical harmonics.
     """,
    articles=['C2DB', Mecholsky2014, Mecholsky2016],
)


def is_band_warped(warping_parameter):
    return abs(warping_parameter) > (1.5 * 1e-3)


def webpanel(result, row, key_descriptions):

    # Make a table from the phonon modes
    data = row.data.get('results-asr.effective_masses.json')
    band_names = ['vbm', 'cbm']
    if data:
        #  unit_cell = data['unit_cell']
        #  reciprocal_unit_cell = data['reciprocal_unit_cell']
        column_list = []
        for band_name in band_names:
            if band_name + '_data' not in data:
                column_list.append(None)
                continue
            band_data = data[band_name + '_data']
            for key in band_data:
                band_data[key] = np.asarray(band_data[key])

            webpanel_table_data = []
            band_warped = is_band_warped(band_data['iems_warping'])

            coords = np.array2string(np.asarray(
                band_data['coords_kbasis']).flatten(), precision=3,
                suppress_small=True, separator=',')
            coords_description = describe_entry(
                'Coordinates',
                'Location of the extremum in crystal coordinates.')

            if band_warped:
                iems_coefficients_k = band_data['iems_coefficients_ks'][:, 0]
                inverse_max_emass = np.min(np.abs(iems_coefficients_k))
                inverse_min_emass = np.max(np.abs(iems_coefficients_k))
            else:
                eigvals = band_data['fit_eigvals']
                max_emass_idx = np.argmin(abs(eigvals))
                min_emass_idx = (1 - max_emass_idx) % 2
                inverse_max_emass = np.abs(eigvals[max_emass_idx]) / 2
                inverse_min_emass = np.abs(eigvals[min_emass_idx]) / 2

            min_emass_description = describe_entry(
                'Min eff. mass',
                'Minimum effective mass of the extremum.'
                ' This corresponds to the inverse curvature of the band in the'
                ' direction of steepest curvature.')

            max_emass_description = describe_entry(
                'Max eff. mass',
                'Maximum effective mass of the extremum.'
                ' This corresponds to the inverse curvature of the band in the'
                ' flattest direction, i.e. the direction of least curvature.')
            if inverse_min_emass > 0:
                min_emass = (1 / inverse_min_emass) * unit_to_electron_mass
                webpanel_table_data.append(
                    (min_emass_description, '%.2f m<sub>0</sub>' % min_emass))
            else:
                webpanel_table_data.append((min_emass_description, 'inf.'))

            if inverse_max_emass > 0:
                max_emass = (1 / inverse_max_emass) * unit_to_electron_mass
                webpanel_table_data.append(
                    (max_emass_description, '%.2f m<sub>0</sub>' % max_emass))
            else:
                webpanel_table_data.append((max_emass_description, 'inf.'))

            m_dos_description = describe_entry(
                'DOS eff. mass',
                'Density of states effective mass as defined in'
                f' {Mecholsky2016}. This parameter equals the effective mass'
                ' of an isotropic extremum which results in the same density'
                ' of states as the present band.')

            if band_warped:
                m_dos = band_data['iems_m_dos'] * unit_to_electron_mass
                if m_dos == np.inf:
                    webpanel_table_data.append((m_dos_description, 'inf.'))
                else:
                    m_dos_as_str = '%.2f' % np.asarray(m_dos)
                    webpanel_table_data.append(
                        (m_dos_description, m_dos_as_str + ' m<sub>0</sub>'))
            else:
                if inverse_max_emass * inverse_min_emass > 0:
                    m_dos = unit_to_electron_mass\
                        / np.sqrt(inverse_max_emass * inverse_min_emass)
                    m_dos_as_str = '%.2f' % m_dos
                    webpanel_table_data.append(
                        (m_dos_description, m_dos_as_str + ' m<sub>0</sub>'))
                else:
                    webpanel_table_data.append((m_dos_description, 'inf.'))

            webpanel_table_data.append((coords_description, coords))

            warping_parameter_description = describe_entry(
                'Warping parameter',
                'Dimensionless warping parameter as defined in'
                f' {Mecholsky2014}.')
            webpanel_table_data.append((warping_parameter_description, '%.3f' %
                                        band_data['iems_warping']))
            X = band_data['contour_kx']
            energy_levels = band_data['barrier_levels']
            dx = X[0, 1] - X[0, 0]
            barrier_R = band_data['barrier_R']
            barrier_height_description = describe_entry(
                'Barrier height',
                'Depth of extremum, quantified by the difference between'
                ' extremum energy and the energy of the smallest potential'
                ' barrier separating this extremum from other extrema.')
            barrier_distance_description = describe_entry(
                'Distance to barrier',
                'Distance in k-space to the nearest potential barrier'
                ' separating this extremum from other extrema.')

            if np.any(np.diff(barrier_R) > 2.9 * dx):
                discont_idx = np.nonzero(
                    np.diff(barrier_R) > 2.9 * dx)[0][0]
                dist_to_barrier = barrier_R[discont_idx]  # size of cbm in 1/Å

            # depth of cbm in meV
                extremum_depth = energy_levels[discont_idx] * 1000

                webpanel_table_data.append((barrier_height_description,
                                            '%.1f meV' % extremum_depth))

                webpanel_table_data.append((barrier_distance_description,
                                            '%.3g Å<sup>-1</sup>'
                                            % dist_to_barrier))

            else:
                webpanel_table_data.append(
                    (barrier_height_description, '> %.1f meV'
                        % (energy_levels.max() * 1000)))

                webpanel_table_data.append(
                    (barrier_distance_description, '> %.1g Å<sup>-1</sup>'
                        % barrier_R.max()))

            column_list.append([fig(band_name + '_contour.png'),
                                {'type': 'table',
                                 'header': ['Property (' + band_name.upper()
                                            + ')', 'Value'],
                                 'rows': webpanel_table_data}])

    else:
        column_list = [None, None]
    # Make the panel
    figure_filenames = [band_name + '_contour.png' for band_name in band_names]
    panel = {'title': describe_entry('Effective masses', panel_description),
             'columns': column_list,
             'plot_descriptions':
                 [{'function': get_figure,
                   'filenames': figure_filenames}],
             'sort': 14}

    return [panel]


@prepare_result
class Result(ASRResult):

    formats = {"ase_webpanel": webpanel}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class GPAW_calculator:

    def __init__(self, params):
        self.n_bands_gs = params['n_bands']
        self.atoms = params['atoms']
        self.gs_calculator = params['gs_calculator']
        self.theta = params['theta']
        self.phi = params['phi']
        self.n_electrons = params['n_electrons']

    def __call__(self, _kpt, band_idx):
        from gpaw import FermiDirac
        from gpaw.spinorbit import soc_eigenstates
        from gpaw import Davidson
        n_bands = max(self.n_bands_gs, self.n_electrons) + 2
        atoms = self.atoms
        calc = self.gs_calculator
        unit_cell = np.array(atoms.cell)
        if len(_kpt.shape) == 1:
            kpt = _kpt.reshape(1, -1)
        else:
            kpt = _kpt.copy()

        if kpt.shape[-1] == 2:
            kpt = np.concatenate((kpt, np.zeros((len(kpt), 1))), axis=1)

        kpt = self.map_to_1BZ(kpt)
        kpt_kbasis = kpt @ unit_cell.T
        calc_result = calc.fixed_density(
            nbands=n_bands,
            symmetry='off',
            kpts=kpt_kbasis,
            eigensolver=Davidson(3),
            maxiter=1000,
            # basis='dzp',
            txt=None,
            occupations=FermiDirac(width=0.05),
            convergence={'bands': 'CBM+3.0'})
        soc = soc_eigenstates(calc_result, theta=self.theta, phi=self.phi,
                              n2=self.n_bands_gs)
        SOC_energies_km = soc.eigenvalues()
        return SOC_energies_km[:, band_idx]

    def map_to_1BZ(self, _kpts):
        reciprocal_unit_cell = np.array(self.atoms.cell.reciprocal())
        unit_cell = np.array(self.atoms.cell)
        kpts_shape = _kpts.shape
        kpts = _kpts.copy()
        if len(kpts_shape) == 1:
            kpts = kpts.reshape(1, -1)
        if kpts_shape[-1] == 2:
            kpts = np.concatenate((kpts, np.zeros((len(kpts), 1))), axis=1)
            truncate_output = True
        else:
            truncate_output = False
        kpts_kbasis = kpts @ unit_cell.T
        # the lines below remove integer factors of reciprocal basis vectors
        # such that the elements of kpts_kbasis lie in the interval [-0.5,0.5)
        kpts_kbasis[:, 0] = (kpts_kbasis[:, 0] + 0.5) % 1 - 0.5
        kpts_kbasis[:, 1] = (kpts_kbasis[:, 1] + 0.5) % 1 - 0.5
        k_ka = kpts_kbasis @ reciprocal_unit_cell
        G1 = reciprocal_unit_cell[0]
        G2 = reciprocal_unit_cell[1]
        NearestReciprocalNeighbors =\
            np.array([np.array([0, 0, 0]), G1, G2, -G1, -G2, G1 - G2,
                      G2 - G1, G1 + G2, -G1 - G2])
        k1BZ_ka = np.zeros(k_ka.shape)

        for i, k in enumerate(k_ka):
            distToNeighbors = np.linalg.norm(k - NearestReciprocalNeighbors,
                                             axis=1)
            closestNeighborIdx = np.argmin(distToNeighbors)
            repetitions = 0
            while closestNeighborIdx != 0 and repetitions < 10:
                repetitions += 1
                k = k - NearestReciprocalNeighbors[closestNeighborIdx]
                distToNeighbors = np.linalg.norm(
                    k - NearestReciprocalNeighbors, axis=1)
                closestNeighborIdx = np.argmin(distToNeighbors)
            if repetitions == 10:
                parprint(
                    'Warning - map to 1BZ did not converge in %d iterations!'
                    % repetitions, flush=True)
            k1BZ_ka[i, 0] = k[0]
            k1BZ_ka[i, 1] = k[1]
        if truncate_output:
            k1BZ_ka = k1BZ_ka[:, :2]
        return k1BZ_ka.reshape(kpts_shape)


@command('asr.effective_masses')
@option('--gspath', help='Path to ground state calculator', type=str)
@option('--atomspath', help='path to file containing atoms structure',
        type=str, default='structure.json')
@option('--calculator', help='Calculator to use for effective mass\
        calculations. If unspecified, ground-state calculator.',
        default=None)
@option('--savefile_name', help='file to save data',
        default='effective_masses.data.json')
@option('--filename_precomputed_data',
        help='file to load data from. useful for restarts',
        default='effective_masses.data.json')
@option('--calculator_args',
        help='extra arguments for calculator',
        default=())
@option('--calculator_kwargs',
        help='extra keyword arguments for calculator',
        default={})
def main(gspath, atomspath='structure.json', calculator=None,
         savefile_name='effective_masses.data.json',
         filename_precomputed_data='effective_masses.data.json',
         *calculator_args, **calculator_kwargs) -> Result:
    from gpaw import GPAW
    from gpaw.spinorbit import soc_eigenstates

    t0 = time.time()

    atoms = ase_read(atomspath)
    assert np.all(atoms.pbc == np.array([True, True, False]))

    gs_calculator = GPAW(gspath)
    atoms.calc = gs_calculator

    data_full = {}
    if filename_precomputed_data is not None\
            and os.path.exists(filename_precomputed_data):
        world.barrier()
        with paropen(filename_precomputed_data, "r") as file:
            precomputed_data = json.load(file)
            data_full.update(precomputed_data)
        for key in data_full:
            if isinstance(data_full[key], list):
                data_full[key] = np.asarray(data_full[key])
            elif isinstance(data_full[key], dict):
                for key_2 in data_full[key]:
                    if isinstance(data_full[key][key_2], list):
                        data_full[key][key_2] \
                            = np.asarray(data_full[key][key_2])
    spin_pol = gs_calculator.get_spin_polarized()
    if spin_pol:
        theta, phi = get_spin_axis()
    else:
        theta = 0
        phi = 0

    if 'obtained_material_parameters' in data_full:
        n_bands = data_full['n_bands']
        unit_cell = data_full['unit_cell']
        reciprocal_unit_cell = data_full['reciprocal_unit_cell']
        n_electrons = data_full['n_electrons']
        cbm_band_idx = data_full['cbm_band_idx']
        vbm_band_idx = data_full['vbm_band_idx']
    else:
        unit_cell = np.array(atoms.cell)[:2, :2]
        reciprocal_unit_cell = np.array(atoms.cell.reciprocal())[:2, :2]
        data_full['unit_cell'] = unit_cell
        data_full['reciprocal_unit_cell'] = reciprocal_unit_cell

        n_electrons = int(gs_calculator.get_number_of_electrons())
        if n_electrons - gs_calculator.get_number_of_electrons() != 0:
            raise ValueError('Structure does not contain an integer number'
                             ' of electrons!')
        data_full['n_electrons'] = n_electrons

        cbm_band_idx = n_electrons
        vbm_band_idx = cbm_band_idx - 1
        data_full['cbm_band_idx'] = cbm_band_idx
        data_full['vbm_band_idx'] = vbm_band_idx

        soc = soc_eigenstates(gs_calculator, theta=theta, phi=phi)
        energies_km = soc.eigenvalues()
        parprint('Calculated GS SOC energies in %.2f seconds' %
                 (time.time() - t0), flush=True)
        t0 = time.time()

        n_bands = gs_calculator.get_number_of_bands()

        data_full['n_bands'] = n_bands
        # check that there is a band gap
        max_valence_energy = np.max(energies_km[:, vbm_band_idx])
        min_conduction_energy = np.min(energies_km[:, cbm_band_idx])
        gap_estimate = min_conduction_energy - max_valence_energy
        gap_threshold = 1e-4  # a gap less than 0.1 meV is not tolerated
        if gap_estimate < gap_threshold:
            assert False, 'no ground state band gap!'

        data_full['obtained_material_parameters'] = True
        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

    # define calculator
    calculator_params = {'n_bands': n_bands,
                         'atoms': atoms,
                         'gs_calculator': gs_calculator,
                         'n_electrons': n_electrons,
                         'theta': theta,
                         'phi': phi}

    calculator = GPAW_calculator(calculator_params)

    get_data(calculator, cbm_band_idx, extremum_type='min',
             data_full=data_full, subdict='cbm_data',
             savefile_name=savefile_name)

    get_data(calculator, vbm_band_idx, extremum_type='max',
             data_full=data_full, subdict='vbm_data',
             savefile_name=savefile_name)

    cbm_energy = data_full['cbm_data']['fit_f0']
    vbm_energy = data_full['vbm_data']['fit_f0']
    band_gap = cbm_energy - vbm_energy
    data_full['band_gap'] = band_gap
    with paropen(savefile_name, "w") as file:
        json.dump(data_full, file, indent=4, cls=NumpyEncoder)

    if world.rank == 0:
        Path(savefile_name).unlink()
    return Result(data_full, strict=False)


def get_data(calculator, band_idx, extremum_type, data_full=None,
             subdict=None, savefile_name=None):
    from gpaw.spinorbit import soc_eigenstates
    from emasses import EmassCalculator, FittedPolynomial
    t0 = time.time()
    EMC = EmassCalculator(calculator, band_idx=band_idx)
    unit_cell = data_full['unit_cell']
    reciprocal_unit_cell = data_full['reciprocal_unit_cell']
    if subdict not in data_full:
        data = {}
        data['completed_steps'] = []
        data_full[subdict] = data
    else:
        data = data_full[subdict]
        data['completed_steps'] = data['completed_steps'].tolist()

    if 'find_extremum' in data['completed_steps']:
        parprint('Reusing coordinates', flush=True)
        coords_cartesian = data['coords_cartesian']
        coords_kbasis = data['coords_kbasis']
        EMC.r0 = coords_cartesian
    else:
        gs_calculator = calculator.gs_calculator
        atoms = calculator.atoms
        spin_pol = gs_calculator.get_spin_polarized()
        if spin_pol:
            theta, phi = get_spin_axis()
        else:
            theta = 0
            phi = 0
        soc = soc_eigenstates(gs_calculator, theta=theta, phi=phi)
        SOC_energies_km = soc.eigenvalues()
        SOC_energies_k = SOC_energies_km[:, band_idx]
        kpts = gs_calculator.get_bz_k_points()[:, :2]
        special_points_3d = atoms.cell.get_bravais_lattice()\
            .get_special_points_array()
        special_points = special_points_3d[special_points_3d[:, 2] == 0][:, :2]
        special_points_cart = special_points @ reciprocal_unit_cell
        energies_special_points = calculator(special_points_cart, band_idx)
        if extremum_type == 'min':
            if np.min(SOC_energies_k) < np.min(energies_special_points):
                arg_estimate = np.argmin(SOC_energies_k)
                x0 = kpts[arg_estimate] @ reciprocal_unit_cell
            else:
                arg_estimate = np.argmin(energies_special_points)
                x0 = special_points_cart[arg_estimate]
        elif extremum_type == 'max':
            if np.max(SOC_energies_k) > np.max(energies_special_points):
                arg_estimate = np.argmax(SOC_energies_k)
                x0 = kpts[arg_estimate] @ reciprocal_unit_cell
            else:
                arg_estimate = np.argmax(energies_special_points)
                x0 = special_points_cart[arg_estimate]
        else:
            raise ValueError('extremum_type must be min or max!')

        coords_cartesian = EMC.find_extremum(x0=x0, xtol=1e-7, ftol=1e-9,
                                             extremum_type=extremum_type)
        coords_cartesian = calculator.map_to_1BZ(coords_cartesian)
        # coords_cartesian = np.array([[coords_cartesian[0],
        #    coords_cartesian[1], 0]])
        coords_kbasis = coords_cartesian @ unit_cell.T

        data['coords_cartesian'] = coords_cartesian
        data['coords_kbasis'] = coords_kbasis
        data['completed_steps'].append('find_extremum')

        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

        parprint('Found extremum in %.2f seconds'
                 % (time.time() - t0), flush=True)
        t0 = time.time()

    # ---------------- Fit polynomial ------------------

    dr0 = min(np.linalg.norm(reciprocal_unit_cell, axis=1)[0],
              np.linalg.norm(reciprocal_unit_cell, axis=1)[1]) / 20
    if 'fit_polynomial' in data['completed_steps']:
        parprint('Reusing polynomial fit', flush=True)
        n_zooms = data['n_zooms']
        fit = FittedPolynomial(r0=data['fit_r0'],
                               f0=data['fit_f0'],
                               gradient=data['fit_gradient'],
                               Hessian=data['fit_Hessian'],
                               xvals=data['fit_xvals'],
                               fvals=data['fit_fvals'],
                               dr=data['fit_dr'])
        EMC.fit = fit
        EMC.n_zooms = n_zooms
    else:
        zoom_result = EMC.zoom_and_fit(r0=coords_cartesian, dr0=dr0)
        fit = zoom_result['fit']
        n_zooms = zoom_result['n_zooms']

        data['n_zooms'] = zoom_result['n_zooms']
        for key in vars(fit):
            new_key = 'fit_' + key
            data[new_key] = vars(fit)[key]
        data['completed_steps'].append('fit_polynomial')

        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

        parprint('Zoomed and fitted polynomial in %.2f seconds' %
                 (time.time() - t0), flush=True)
        t0 = time.time()
        parprint('Zoomed %d times' % n_zooms, flush=True)

    # - get IEMS -----
    if 'get_iems' in data['completed_steps']:
        parprint('Reusing iems', flush=True)
        iems = {}
        iems['r2'] = data['iems_r2']

    else:
        iems = EMC.get_iems_2d(fit, N=100)

        for key in iems:
            new_key = 'iems_' + key
            data[new_key] = iems[key]
        data['completed_steps'].append('get_iems')

        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

    # ----------- Surface plot ------------

    side_length = max(np.linalg.norm(reciprocal_unit_cell, axis=1)[
        0] / 8, np.linalg.norm(reciprocal_unit_cell, axis=1)[1] / 8,
        2.1 * iems['r2'])

    if 'get_contour' in data['completed_steps']:
        parprint('Reusing contour', flush=True)
        contour = {}
        contour['contour_kx'] = data['contour_kx']
        contour['contour_ky'] = data['contour_ky']
        contour['contour_energies'] = data['contour_energies']

    else:

        contour = EMC.get_contour_2d(fit, side_length=side_length)
        data['contour_kx'] = contour['contour_kx']
        data['contour_ky'] = contour['contour_ky']
        data['contour_energies'] = contour['contour_energies']

        data['completed_steps'].append('get_contour')
        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

    # ---------- Find barrier ----------------
    if 'get_barrier' in data['completed_steps']:
        print('Reusing barrier', flush=True)
        # no need to do anything here, as this is not used subsequently
        # in the code
    else:
        X = contour['contour_kx']
        Y = contour['contour_ky']
        if extremum_type == 'min':
            Z = contour['contour_energies']
        elif extremum_type == 'max':
            Z = - contour['contour_energies']
        R = np.sqrt(X**2 + Y**2)
        barrier_levels = EMC.find_barrier(R, Z, levels=np.linspace(
            0, Z.max() - Z.min(), 3500), R_max=X.max())

        data['barrier_R'] = barrier_levels['R']
        data['barrier_R_idx'] = barrier_levels['R_idx']
        data['barrier_levels'] = barrier_levels['levels']
        data['below_barrier_list'] = barrier_levels['Flooded_list']
        data['below_barrier_kk'] = barrier_levels['Flooded']

        data['completed_steps'].append('get_barrier')
        with paropen(savefile_name, "w") as file:
            json.dump(data_full, file, indent=4, cls=NumpyEncoder)

    parprint('Finished after an additional %.2f seconds' %
             (time.time() - t0), flush=True)

    return


def get_figure(row, *filenames):
    data = row.data.get('results-asr.effective_masses.json')
    band_names = ['vbm', 'cbm']  # hard coded band-names. change in future
    for i, filename in enumerate(filenames):
        band_name = band_names[i]  # remove _contour.png from name
        band_data = data[band_name + '_data']
        for key in band_data:
            band_data[key] = np.asarray(band_data[key])

        X = band_data['contour_kx']
        Y = band_data['contour_ky']
        f0 = band_data['fit_f0']
        Z = band_data['contour_energies'] - f0
        phi = band_data['iems_phi']
        iems_coefficients_k = band_data['iems_coefficients_ks'][:, 0]

        band_warped = is_band_warped(band_data['iems_warping'])
        if band_warped:
            max_emass_angle = phi[np.argmin(np.abs(iems_coefficients_k))]
            max_emass_direction = np.array(
                [np.cos(max_emass_angle), np.sin(max_emass_angle)])
            min_emass_angle\
                = phi[np.argmax(np.abs(iems_coefficients_k))]
            min_emass_direction = np.array(
                [np.cos(min_emass_angle), np.sin(min_emass_angle)])
        else:
            eigvals = band_data['fit_eigvals']
            eigvecs = band_data['fit_eigvecs']
            max_emass_idx = np.argmin(abs(eigvals))
            min_emass_idx = (1 - max_emass_idx) % 2
            max_emass_direction = eigvecs[:, max_emass_idx]
            min_emass_direction = eigvecs[:, min_emass_idx]

        line_radius = np.sqrt(X**2 + Y**2).max() / 2
        line = np.linspace(0, line_radius, 101)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, Z, cmap='viridis',
                    levels=80, vmin=Z.min(), vmax=Z.max())

        diff = Z.max() - Z.min()
        if band_name == 'cbm':
            extra_contours = Z.min()\
                + (np.linspace(0, np.sqrt(diff), 10)[1:-1])**2
            ax.contour(X, Y, Z, cmap='viridis', levels=extra_contours,
                       vmin=Z.min() - 0.1 * diff, vmax=Z.max() + 0.1 * diff)
        elif band_name == 'vbm':
            extra_contours = np.flip(
                Z.max() - (np.linspace(0, np.sqrt(diff), 10)[1:-1])**2)
            ax.contour(X, Y, Z, cmap='viridis', levels=extra_contours,
                       vmin=Z.min() - 0.1 * diff, vmax=Z.max() + 0.1 * diff)

        ax.set_xlabel(r'$k_x$ / Å$^{-1}$')
        ax.set_ylabel(r'$k_y$ / Å$^{-1}$')
        cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis', norm=pltnorm(
            1000 * Z.min(), 1000 * Z.max())), ax=ax)
        cbar.set_label(r'$(E - E_0)$ / meV')

        # add circles to contour plot
        ax.plot(line * max_emass_direction[0], line
                * max_emass_direction[1], color='tab:green', ls='dashed',
                label='Max eff. mass direction')
        ax.plot(line * min_emass_direction[0], line
                * min_emass_direction[1], color='tab:orange', ls='dashed',
                label='Min eff. mass direction')

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        if band_name == 'cbm':
            plt.title('Conduction band minimum (CBM)')
        elif band_name == 'vbm':
            plt.title('Valence band maximum (VBM)')
        else:
            plt.title(band_name)

        plt.tight_layout()

        if len(ax.get_xticks()) >= 7:
            ax.set_xticklabels(np.round(ax.get_xticks(), 4), rotation=15)
        ax.legend()

        plt.savefig(filename)


if __name__ == '__main__':
    main.cli()
