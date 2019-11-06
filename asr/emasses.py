from asr.core import command, option


@command('asr.emasses',
         requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         creates=['dense_k.gpw'])
@option('--gpwfilename', type=str,
        help='GS filename')
@option('--kptdensity', type=int,
        help='kpt density')
@option('--emptybands', type=int,
        help='Number of empty bands')
def densify_full_k_grid(gpwfilename='gs.gpw', kptdensity=12,
                        emptybands=20):
    from gpaw import GPAW
    from asr.utils.kpts import get_kpts_size
    calc = GPAW(gpwfilename, txt=None)
    spinpol = calc.get_spin_polarized()

    kpts = get_kpts_size(atoms=calc.atoms, density=kptdensity)
    convbands = emptybands // 2
    calc.set(nbands=-emptybands,
             txt='dense_k.txt',
             fixdensity=True,
             kpts=kpts,
             convergence={'bands': -convbands})

    if spinpol:
        calc.set(symmetry='off')

    calc.get_potential_energy()
    calc.write('dense_k.gpw')


@command('asr.emasses',
         requires=['dense_k.gpw', 'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.emasses@densify_full_k_grid', 'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         creates=['em_circle_vb_nosoc.gpw', 'em_circle_cb_nosoc.gpw',
                  'em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw'])
@option('--gpwfilename', type=str,
        help='GS Filename')
def refine(gpwfilename='dense_k.gpw'):
    '''
    Take a bandstructure and calculate more kpts around
    the vbm and cbm
    '''
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    import os.path
    socs = [True, False]

    for soc in socs:
        theta, phi = get_spin_axis()
        eigenvalues, efermi = gpw2eigs(gpw=gpwfilename, soc=soc,
                                       theta=theta, phi=phi)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'

            if not gap > 0:
                from gpaw import GPAW
                calc = GPAW(gpwfilename, txt=None)
                calc.write(gpw2)
                continue
            if os.path.exists(gpw2):
                continue
            nonsc_sphere(gpw=gpwfilename, soc=soc, bandtype=bt)


def get_name(soc, bt):
    return 'em_circle_{}_{}'.format(bt, ['nosoc', 'soc'][soc])


def nonsc_sphere(gpw='gs.gpw', soc=False, bandtype=None):
    """non sc calculation based for kpts in a sphere around the
        valence band maximum and conduction band minimum.
        writes the files:
            em_circle_vb_soc.gpw
            em_circle_cb_soc.gpw
            em_circle_vb_nosoc.gpw
            em_circle_cb_nosoc.gpw
        Parameters:
            gpw: str
                gpw filename
            soc: bool
                spinorbit coupling
            bandtype: None or 'cb' or 'vb'
                which bandtype do we do calculations for, if None is done for
                for both cb and vb

    """
    from gpaw import GPAW
    import numpy as np
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()

    # Check that 1D: Only z-axis, 2D: Only x- and y-axis
    if ndim == 1:
        pbc = calc.atoms.pbc
        assert not pbc[0] and not pbc[1] and pbc[2]
    elif ndim == 2:
        pbc = calc.atoms.pbc
        assert pbc[0] and pbc[1] and not pbc[2]

    k_kc = calc.get_ibz_k_points()
    cell_cv = calc.atoms.get_cell()

    kcirc_kc = kptsinsphere(cell_cv, dimensionality=ndim)

    theta, phi = get_spin_axis()
    e_skn, efermi = gpw2eigs(gpw, soc=soc, theta=theta, phi=phi)
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]

    _, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn, efermi=efermi,
                                            output=None)

    k1_c = k_kc[k1]
    k2_c = k_kc[k2]

    bandtypes, ks = get_bt_ks(bandtype, k1_c, k2_c)

    for bt, k_c in zip(bandtypes, ks):
        name = get_name(soc=soc, bt=bt)
        calc.set(kpts=kcirc_kc + k_c,
                 symmetry='off',
                 txt=name + '.txt')
        atoms = calc.get_atoms()
        atoms.get_potential_energy()
        calc.write(name + '.gpw')


def kptsinsphere(cell_cv, npoints=9, erange=1e-3, m=1.0, dimensionality=3):
    import numpy as np
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert

    a = np.linspace(-1, 1, npoints)
    X, Y, Z = np.meshgrid(a, a, a)

    na = np.logical_and
    if dimensionality == 2:
        indices = na(X**2 + Y**2 <= 1.0, Z == 0)
    elif dimensionality == 1:
        indices = na(Z**2 <= 1.0, na(X == 0, Y == 0))
    else:
        indices = X**2 + Y**2 + Z**2 <= 1.0

    x, y, z = X[indices], Y[indices], Z[indices]
    kpts_kv = np.vstack([x, y, z]).T
    kr = np.sqrt(2 * m * erange / Hartree)
    kpts_kv *= kr
    kpts_kv /= Bohr
    kpts_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=kpts_kv)
    return kpts_kc

    # print(kpts_kv)

    # a = np.linspace(-1, 1, npoints)
    # X, Y, Z = np.meshgrid(a, a, a)
    # indices = X**2 + Y**2 + Z**2 <= 1.0
    # sh = X.shape
    # x, y, z = X[indices], Y[indices], Z[indices]
    # kpts_kv = np.vstack([x, y * yfactor, z * zfactor]).T


def get_bt_ks(bandtype, k1_c, k2_c):
    if bandtype is None:
        bandtypes = ('vb', 'cb')
        ks = (k1_c, k2_c)
    elif bandtype == 'vb':
        bandtypes = ('vb',)
        ks = (k1_c, )
    elif bandtype == 'cb':
        bandtypes = ('cb', )
        ks = (k2_c, )
    return bandtypes, ks


def get_ndims_from_row(row):
    import numpy as np
    results = row.data.get('results-asr.emasses.json')

    for k in results.keys():
        if '(' in k and ')' in k:
            count_dirs = 0
            for k2 in results[k].keys():
                if 'effmass' in k2:
                    value = results[k][k2]
                    if not (value is None) and not np.isnan(value):
                        count_dirs += 1
            break
    return count_dirs


def convert_key_to_tuple(key):
    k = key.replace("(", "").replace(")", "")
    ks = k.split(",")
    ks = [k.strip() for k in ks]
    ks = [int(k) for k in ks]
    return tuple(ks)


def get_emass_dict_from_row(row):
    import numpy as np
    results = row.data.get('results-asr.emasses.json')

    cb_indices = []
    vb_indices = []
    for k in results.keys():
        if '(' in k and ')' in k:
            for k2 in results[k].keys():
                if 'nosoc' in k2:
                    break

                if 'vb_soc_effmass' in k2:
                    vb_indices.append(k)
                    break
                elif 'cb_soc_effmass' in k2:
                    cb_indices.append(k)
                    break

    cb_indices = [(k, convert_key_to_tuple(k)[1]) for k in cb_indices]
    vb_indices = [(k, convert_key_to_tuple(k)[1]) for k in vb_indices]

    ordered_cb_indices = sorted(cb_indices, key=lambda el: el[1])
    ordered_vb_indices = sorted(vb_indices, key=lambda el: -el[1])

    def get_the_dict(ordered_indices, name, offset_sym):
        my_dict = {}
        for offset_num, (key, band_number) in enumerate(ordered_indices):
            data = results[key]
            direction = 0
            for k in data.keys():
                if 'effmass' in k:
                    mass = data[k]
                    if mass is not None and not np.isnan(mass):
                        direction += 1
                        mass_str = str(round(abs(mass) * 100) / 100)
                        if offset_num == 0:
                            my_dict[f'{name}, direction {direction}'] = \
                                '{}'.format(mass_str)
                        else:
                            my_dict['{} {} {}, direction {}'.format(
                                name, offset_sym,
                                offset_num, direction)] = mass_str
        return my_dict

    electron_dict = get_the_dict(ordered_cb_indices, 'CB', '+')
    hole_dict = get_the_dict(ordered_vb_indices, 'VB', '-')

    return electron_dict, hole_dict


def dummyfunc(row, *args):
    return None


def make_the_plots(row, *args):
    from ase.dft.kpoints import kpoint_convert, labels_from_kpts
    from ase.units import Bohr, Ha
    import matplotlib.pyplot as plt
    import numpy as np
    from asr.browser import fig as asrfig

    results = row.data.get('results-asr.emasses.json')

    columns = []
    cb_fnames = []
    vb_fnames = []

    vb_indices = []
    cb_indices = []

    for spin_band_str, data in results.items():
        if '__' in spin_band_str:
            continue
        is_w_soc = check_soc(data)
        if not is_w_soc:
            continue
        for k in data.keys():
            if 'effmass' in k and 'vb' in k:
                vb_indices.append(spin_band_str)
                break
            if 'effmass' in k and 'cb' in k:
                cb_indices.append(spin_band_str)
                break

    cb_masses = {}
    vb_masses = {}

    for cb_key in cb_indices:
        data = results[cb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(cb_key)
        cb_masses[tuple_key] = masses

    for vb_key in vb_indices:
        data = results[vb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(vb_key)
        vb_masses[tuple_key] = masses

    erange = 0.05

    def get_range(mass, _erange):
        return (2 * mass * _erange / Ha) ** 0.5 / Bohr

    plt_count = 0
    for direction in range(3):
        y1 = None
        y2 = None
        my_range = None
        # CB plots
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [1]})

        should_plot = True
        for cb_key in cb_indices:
            cb_tuple = convert_key_to_tuple(cb_key)
            # Save something
            data = results[cb_key]
            fit_data_list = data['cb_soc_bzcuts']
            if direction >= len(fit_data_list):
                should_plot = False
                continue

            mass = cb_masses[cb_tuple][-1 - direction]
            fit_data = fit_data_list[direction]
            ks = fit_data['kpts_kc']
            cell_cv = fit_data['cell_cv']
            xk, _, _ = labels_from_kpts(kpts=ks, cell=cell_cv)
            xk -= xk[-1] / 2
            kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
            kpts_kv *= Bohr

            e_km = fit_data['e_dft_km']
            sz_km = fit_data['sz_dft_km']
            if e_km.ndim == 2:
                e_k = e_km[:, cb_tuple[1]]
            else:
                e_k = e_km[0, :, cb_tuple[1]]
            if sz_km.ndim == 2:
                sz_k = sz_km[:, cb_tuple[1]]
            else:
                sz_k = sz_km[0, :, cb_tuple[1]]

            emodel_k = (xk * Bohr) ** 2 / (2 * mass) * Ha
            emodel_k += np.min(e_k) - np.min(emodel_k)

            things = axes.scatter(xk, e_k, c=sz_k, vmin=-1, vmax=1)
            axes.plot(xk, emodel_k, c='r', ls='--')

            if y1 is None or y2 is None or my_range is None:
                y1 = np.min(emodel_k) - erange * 0.25
                y2 = np.min(emodel_k) + erange * 0.75
                axes.set_ylim(y1, y2)

                my_range = get_range(mass, erange)
                axes.set_xlim(-my_range, my_range)

                cbar = fig.colorbar(things, ax=axes)
                cbar.set_label(r'$\langle S_z \rangle$')
                cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
                cbar.update_ticks()

            fig.tight_layout()
            plt.tight_layout()
        if should_plot:
            fname = args[plt_count]
            plt.savefig(fname)
            plt.close()

        # axes.clear()
        # VB plots
        y1 = None
        y2 = None
        my_range = None
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [1]})

        for vb_key in vb_indices:
            # Save something
            vb_tuple = convert_key_to_tuple(vb_key)
            data = results[vb_key]
            fit_data_list = data['vb_soc_bzcuts']
            if direction >= len(fit_data_list):
                continue

            mass = vb_masses[vb_tuple][direction]
            fit_data = fit_data_list[direction]
            ks = fit_data['kpts_kc']
            cell_cv = fit_data['cell_cv']
            xk2, _, _ = labels_from_kpts(kpts=ks, cell=cell_cv)
            xk2 -= xk2[-1] / 2
            kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
            kpts_kv *= Bohr

            e_km = fit_data['e_dft_km']
            sz_km = fit_data['sz_dft_km']
            if e_km.ndim == 2:
                e_k = e_km[:, vb_tuple[1]]
            else:
                e_k = e_km[0, :, vb_tuple[1]]
            if sz_km.ndim == 2:
                sz_k = sz_km[:, vb_tuple[1]]
            else:
                sz_k = sz_km[0, :, vb_tuple[1]]

            emodel_k = (xk2 * Bohr) ** 2 / (2 * mass) * Ha
            emodel_k += np.max(e_k) - np.max(emodel_k)

            things = axes.scatter(xk2, e_k, c=sz_k, vmin=-1, vmax=1)
            axes.plot(xk2, emodel_k, c='r', ls='--')

            if y1 is None or y2 is None or my_range is None:
                y1 = np.max(emodel_k) - erange * 0.75
                y2 = np.max(emodel_k) + erange * 0.25
                axes.set_ylim(y1, y2)

                my_range = get_range(abs(mass), erange)
                axes.set_xlim(-my_range, my_range)

                cbar = fig.colorbar(things, ax=axes)
                cbar.set_label(r'$\langle S_z \rangle$')
                cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
                cbar.update_ticks()

            fig.tight_layout()
            plt.tight_layout()
        if should_plot:
            nplts = len(args)
            fname = args[plt_count + nplts // 2]
            plt.savefig(fname)
            plt.close()
        plt_count += 1
    # Make final column with table of numerical vals

    # Make table

    # Algo:
    # Loop through directions, each direction is a column
    # For direction i, loop through cbs and plot on fig
    # -- Plot also quadratic fit from curvature/effective mass value
    # For direction i, loop through vbs and plot on fig
    # Make a final column containing a table with the numerical values
    # for the effective masses
    assert len(cb_fnames) == len(vb_fnames), \
        'Num cb plots: {}\nNum vb plots: {}'.format(
        len(cb_fnames), len(vb_fnames))

    num_cols = len(cb_fnames)

    for j in range(num_cols):
        cb_fname = cb_fnames[j]
        vb_fname = vb_fnames[j]
        col = [asrfig(cb_fname), asrfig(vb_fname)]
        columns.append(col)

    return


def custom_table(values_dict, title):
    table = {'type': 'table',
             'header': [title, 'Value']}

    rows = []
    for k in values_dict.keys():
        rows.append((k, values_dict[k]))

    table['rows'] = rows
    print("")
    print("")
    print("")
    print("ROWS = ", rows)
    print("")
    print("")
    print("")

    return table


def webpanel(row, key_descriptions):
    from asr.browser import table

    columns, fnames = create_columns_fnames(row)

    electron_dict, hole_dict = get_emass_dict_from_row(row)

    electron_table = custom_table(electron_dict, 'Electron effective mass')
    hole_table = custom_table(hole_dict, 'Hole effective mass')
    electron_table2 = table(electron_dict, 'Electron effective mass', list(
        electron_dict.keys()), key_descriptions)
    # kd={k: ("", k, "") for k in electron_dict.keys()})
    # hole_table = table(hole_dict, 'Hole effective mass',
    # list(hole_dict.keys()),
    # key_descriptions)#kd={k: ("", k, "") for k in electron_dic
    # t.keys()})

    print("MY TABLE: ", electron_table)

    print("")
    print("")
    print("")
    print("STUPID TABLE: ", electron_table2)
    print("")
    print("")
    print("")

    table_col = [electron_table, hole_table]

    columns.append(table_col)

    for col in columns:
        assert isinstance(col, list)
        print("ASSERTIVE")
    panel = {'title': 'Effective masses (PBE)',
             'columns': columns,
             'plot_descriptions':
             [{'function': make_the_plots,
               'filenames': fnames
               }],
             'sort': 10}
    print('emasses fnames', fnames)
    print('In emasses webpanel', columns[-1])
    print("")
    print("")
    print("")
    print("LEN OF COLUMNS: ", len(panel['columns']))
    print("")
    print("")
    return [panel]


def create_columns_fnames(row):
    from asr.browser import fig as asrfig

    results = row.data.get('results-asr.emasses.json')

    columns = []
    cb_fnames = []
    vb_fnames = []

    vb_indices = []
    cb_indices = []

    for spin_band_str, data in results.items():
        if '__' in spin_band_str:
            continue
        is_w_soc = check_soc(data)
        if not is_w_soc:
            continue
        for k in data.keys():
            if 'effmass' in k and 'vb' in k:
                vb_indices.append(spin_band_str)
                break
            if 'effmass' in k and 'cb' in k:
                cb_indices.append(spin_band_str)
                break

    cb_masses = {}
    vb_masses = {}

    for cb_key in cb_indices:
        data = results[cb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(cb_key)
        cb_masses[tuple_key] = masses

    for vb_key in vb_indices:
        data = results[vb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(vb_key)
        vb_masses[tuple_key] = masses

    for direction in range(3):
        should_plot = True
        for cb_key in cb_indices:
            data = results[cb_key]
            fit_data_list = data['cb_soc_bzcuts']
            if direction >= len(fit_data_list):
                should_plot = False
                continue
        if should_plot:
            fname = 'cb_dir_{}.png'.format(direction)
            cb_fnames.append(fname)

            fname = 'vb_dir_{}.png'.format(direction)
            vb_fnames.append(fname)

    assert len(cb_fnames) == len(vb_fnames), \
        'Num cb plots: {}\nNum vb plots: {}'.format(
        len(cb_fnames), len(vb_fnames))

    num_cols = len(cb_fnames)

    for j in range(num_cols):
        cb_fname = cb_fnames[j]
        vb_fname = vb_fnames[j]
        col = [asrfig(cb_fname), asrfig(vb_fname)]
        columns.append(col)

    return columns, cb_fnames + vb_fnames

    # from ase.dft.kpoints import kpoint_convert, labels_from_kpts
    # from ase.units import Bohr, Ha
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from asr.browser import fig as asrfig
    # from asr.browser import table

    # results = row.data.get('results-asr.emasses.json')

    # columns = []
    # cb_fnames = []
    # vb_fnames = []

    # vb_indices = []
    # cb_indices = []

    # for spin_band_str, data in results.items():
    #     if '__' in spin_band_str:
    #         continue
    #     is_w_soc = check_soc(data)
    #     if not is_w_soc:
    #         continue
    #     for k in data.keys():
    #         if 'effmass' in k and 'vb' in k:
    #             vb_indices.append(spin_band_str)
    #             break
    #         if 'effmass' in k and 'cb' in k:
    #             cb_indices.append(spin_band_str)
    #             break

    # cb_masses = {}
    # vb_masses = {}

    # for cb_key in cb_indices:
    #     data = results[cb_key]
    #     masses = []
    #     for k in data.keys():
    #         if 'effmass' in k:
    #             masses.append(data[k])
    #     tuple_key = convert_key_to_tuple(cb_key)
    #     cb_masses[tuple_key] = masses

    # for vb_key in vb_indices:
    #     data = results[vb_key]
    #     masses = []
    #     for k in data.keys():
    #         if 'effmass' in k:
    #             masses.append(data[k])
    #     tuple_key = convert_key_to_tuple(vb_key)
    #     vb_masses[tuple_key] = masses

    # erange = 0.05
    # def get_range(mass, _erange):
    #     return (2 * mass *_erange / Ha) ** 0.5 / Bohr

    # for direction in range(3):
    #     y1 = None
    #     y2 = None
    #     my_range = None
    #     # CB plots
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
    #                              sharey=True,
    #                              gridspec_kw={'width_ratios': [1]})

    #     should_plot = True
    #     for cb_key in cb_indices:
    #         cb_tuple = convert_key_to_tuple(cb_key)
    #         # Save something
    #         data = results[cb_key]
    #         fit_data_list = data['cb_soc_bzcuts']
    #         if direction >= len(fit_data_list):
    #             should_plot = False
    #             continue

    #         mass = cb_masses[cb_tuple][-1 - direction]
    #         fit_data = fit_data_list[direction]
    #         ks = fit_data['kpts_kc']
    #         cell_cv = fit_data['cell_cv']
    #         xk, _, _ = labels_from_kpts(kpts=ks, cell=cell_cv)
    #         xk -= xk[-1] / 2
    #         kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
    #         kpts_kv *= Bohr

    #         e_km = fit_data['e_dft_km']
    #         sz_km = fit_data['sz_dft_km']
    #         if e_km.ndim == 2:
    #             e_k = e_km[:, cb_tuple[1]]
    #         else:
    #             e_k = e_km[0, :, cb_tuple[1]]
    #         if sz_km.ndim == 2:
    #             sz_k = sz_km[:, cb_tuple[1]]
    #         else:
    #             sz_k = sz_km[0, :, cb_tuple[1]]

    #         emodel_k = (xk * Bohr) ** 2 / (2 * mass) * Ha
    #         emodel_k += np.min(e_k) - np.min(emodel_k)

    #         things = axes.scatter(xk, e_k, c=sz_k, vmin=-1, vmax=1)
    #         axes.plot(xk, emodel_k, c='r', ls='--')

    #         if y1 is None or y2 is None or my_range is None:
    #             y1 = np.min(emodel_k) - erange * 0.25
    #             y2 = np.min(emodel_k) + erange * 0.75
    #             axes.set_ylim(y1, y2)

    #             my_range = get_range(mass, erange)
    #             axes.set_xlim(-my_range, my_range)

    #             cbar = fig.colorbar(things, ax=axes)
    #             cbar.set_label(r'$\langle S_z \rangle$')
    #             cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    #             cbar.update_ticks()

    #         fig.tight_layout()
    #         plt.tight_layout()
    #     if should_plot:
    #         fname = 'cb_dir_{}.png'.format(direction)
    #         cb_fnames.append(fname)
    #         plt.savefig(fname)
    #         plt.close()

    #     # axes.clear()
    #     # VB plots
    #     y1 = None
    #     y2 = None
    #     my_range = None
    #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
    #                              sharey=True,
    #                              gridspec_kw={'width_ratios': [1]})

    #     for vb_key in vb_indices:
    #         # Save something
    #         vb_tuple = convert_key_to_tuple(vb_key)
    #         data = results[vb_key]
    #         fit_data_list = data['vb_soc_bzcuts']
    #         if direction >= len(fit_data_list):
    #             continue

    #         mass = vb_masses[vb_tuple][direction]
    #         fit_data = fit_data_list[direction]
    #         ks = fit_data['kpts_kc']
    #         cell_cv = fit_data['cell_cv']
    #         xk2, _, _ = labels_from_kpts(kpts=ks, cell=cell_cv)
    #         xk2 -= xk2[-1] / 2
    #         kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
    #         kpts_kv *= Bohr

    #         e_km = fit_data['e_dft_km']
    #         sz_km = fit_data['sz_dft_km']
    #         if e_km.ndim == 2:
    #             e_k = e_km[:, vb_tuple[1]]
    #         else:
    #             e_k = e_km[0, :, vb_tuple[1]]
    #         if sz_km.ndim == 2:
    #             sz_k = sz_km[:, vb_tuple[1]]
    #         else:
    #             sz_k = sz_km[0, :, vb_tuple[1]]

    #         emodel_k = (xk2 * Bohr) ** 2 / (2 * mass) * Ha
    #         emodel_k += np.max(e_k) - np.max(emodel_k)

    #         things = axes.scatter(xk2, e_k, c=sz_k, vmin=-1, vmax=1)
    #         axes.plot(xk2, emodel_k, c='r', ls='--')

    #         if y1 is None or y2 is None or my_range is None:
    #             y1 = np.max(emodel_k) - erange * 0.75
    #             y2 = np.max(emodel_k) + erange * 0.25
    #             axes.set_ylim(y1, y2)

    #             my_range = get_range(abs(mass), erange)
    #             axes.set_xlim(-my_range, my_range)

    #             cbar = fig.colorbar(things, ax=axes)
    #             cbar.set_label(r'$\langle S_z \rangle$')
    #             cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    #             cbar.update_ticks()

    #         fig.tight_layout()
    #         plt.tight_layout()
    #     if should_plot:
    #         fname = 'vb_dir_{}.png'.format(direction)
    #         vb_fnames.append(fname)
    #         plt.savefig(fname)
    #         plt.close()

    # # Make final column with table of numerical vals

    # # Make table

    # # Algo:
    # # Loop through directions, each direction is a column
    # # For direction i, loop through cbs and plot on fig
    # # -- Plot also quadratic fit from curvature/effective mass value
    # # For direction i, loop through vbs and plot on fig
    # # Make a final column containing a table with the numerical values
    # # for the effective masses
    # assert len(cb_fnames) == len(vb_fnames), \
    #     'Num cb plots: {}\nNum vb plots: {}'.format(len(cb_fnames),
    #                                                 len(vb_fnames))

    # num_cols = len(cb_fnames)

    # for j in range(num_cols):
    #     cb_fname = cb_fnames[j]
    #     vb_fname = vb_fnames[j]
    #     col = [asrfig(cb_fname), asrfig(vb_fname)]
    #     columns.append(col)

    # return columns, cb_fnames + vb_fnames

    # for spin_band_str, data in results.items():
    #     data = results[spin_band_str]

    #     is_w_soc = check_soc(data)
    #     if not is_w_soc:
    #         continue

    #     mass = None
    #     for k in data.keys():
    #         if 'effmass' in k and 'dir' + str(direction) in k:
    #             mass = data[k]

    #     if mass is None or np.isnan(mass):
    #         continue

    #     # Create a column
    #     # A column should consist of the plots of the eigenvalues
    #     # and the quadratic fit for all the vbs/cbs for a given dir
    #     # lastly we should have a column with the numerical values
    #     #        panel = {'title': 'Effective masses

    # return None


def check_soc(spin_band_dict):
    for k in spin_band_dict.keys():
        if 'effmass' in k and 'nosoc' in k:
            return False

    return True


@command('asr.emasses',
         requires=['em_circle_vb_nosoc.gpw', 'em_circle_cb_nosoc.gpw',
                   'em_circle_vb_soc.gpw', 'em_circle_cb_soc.gpw',
                   'dense_k.gpw', 'results-asr.structureinfo.json',
                   'results-asr.magnetic_anisotropy.json'],
         dependencies=['asr.emasses@refine', 'asr.gs@calculate',
                       'asr.emasses@densify_full_k_grid',
                       'asr.structureinfo',
                       'asr.magnetic_anisotropy'],
         webpanel=webpanel)
@option('--gpwfilename', type=str,
        help='GS Filename')
def main(gpwfilename='dense_k.gpw'):
    from asr.utils.gpw2eigs import gpw2eigs
    from ase.dft.bandgap import bandgap
    from asr.magnetic_anisotropy import get_spin_axis
    import traceback
    socs = [True, False]

    good_results = {}
    for soc in socs:
        theta, phi = get_spin_axis()
        eigenvalues, efermi = gpw2eigs(gpw=gpwfilename, soc=soc,
                                       theta=theta, phi=phi)
        gap, _, _ = bandgap(eigenvalues=eigenvalues, efermi=efermi,
                            output=None)
        if not gap > 0:
            continue
        for bt in ['vb', 'cb']:
            name = get_name(soc=soc, bt=bt)
            gpw2 = name + '.gpw'
            try:
                masses = embands(gpw2,
                                 soc=soc,
                                 bandtype=bt,
                                 efermi=efermi)

                # This function modifies the last argument
                unpack_masses(masses, soc, bt, good_results)
            except ValueError:
                tb = traceback.format_exc()
                print(gpw2 + ':\n' + '=' * len(gpw2) + '\n', tb)
            else:
                _savemass(soc=soc, bt=bt, mass=masses)

    bands = 3
    kdescs = {}
    for j in range(bands):
        if j == 0:
            for k in range(3):
                kdescs['CB, direction {}'.format(k)] = \
                    'KVP: CB, direction {}'.format(
                    k) + r' [$\mathrm{m_e}$]'
                kdescs['VB, direction {}'.format(k)] = \
                    'KVP: VB, direction {}'.format(
                    k) + r' [$\mathrm{m_e}$]'
        else:
            for k in range(3):
                kdescs['CB + {}, direction {}'.format(j, k)] = \
                    'KVP: CB + {}, direction {}'.format(j, k) \
                    + r' [$\mathrm{m_e}$]'
                kdescs['VB - {}, direction {}'.format(j, k)] = \
                    'KVP: VB - {}, direction {}'.format(j, k) + \
                    r' [$\mathrm{m_e}$]'

    good_results['__key_descriptions__'] = kdescs

    return good_results


def unpack_masses(masses, soc, bt, results_dict):
    # Structure of 'masses' object:
    # There is an 'indices' key which tells you at which spin-k indices
    # effective masses have been calculated
    # At a given index there is saved the dict returned by the em function:
    # out = dict(mass_u=masses, eigenvectors_vu=vecs,
    #            ke_v=kmax,
    #            c=c,
    #            r=r)
    # We want to flatten this structure so that results_dict contains
    # the masses directly
    for ind in masses['indices']:
        out_dict = masses[ind]
        index = str(ind)
        socpre = 'soc' if soc else 'nosoc'
        prefix = bt + '_' + socpre + '_'

        results_dict[index] = {}

        results_dict[index][prefix + 'effmass_dir1'] = out_dict['mass_u'][0]
        results_dict[index][prefix + 'effmass_dir2'] = out_dict['mass_u'][1]
        results_dict[index][prefix + 'effmass_dir3'] = out_dict['mass_u'][2]
        vecs = out_dict['eigenvectors_vu']
        results_dict[index][prefix + 'eigenvectors_vdir1'] = vecs[:, 0]
        results_dict[index][prefix + 'eigenvectors_vdir2'] = vecs[:, 1]
        results_dict[index][prefix + 'eigenvectors_vdir3'] = vecs[:, 2]
        results_dict[index][prefix + 'spin'] = ind[0]
        results_dict[index][prefix + 'bandindex'] = ind[1]
        results_dict[index][prefix + 'kpt_v'] = out_dict['ke_v']
        results_dict[index][prefix + 'fitcoeff'] = out_dict['c']
        results_dict[index][prefix + '2ndOrderFit'] = out_dict['c2']
        results_dict[index][prefix + 'mass_u'] = out_dict['mass_u']
        results_dict[index][prefix + 'bzcuts'] = out_dict['bs_along_emasses']
        results_dict[index][prefix + 'fitkpts_kv'] = out_dict['fitkpts_kv']
        results_dict[index][prefix + 'fite_k'] = out_dict['fite_k']


def embands(gpw, soc, bandtype, efermi=None, delta=0.1):
    """effective masses for bands within delta of extrema
    Parameters:
        gpw: str
            name of gpw filename
        soc: bool
            include spin-orbit coupling
        bandtype: 'vb' or 'cb'
            type of band
        efermi: float, optional
            fermi level (takes it from gpw if None)
        delta: float, optional
            bands within this value (in eV) is included in the em fit
            default is 0.1 eV
    """
    from gpaw import GPAW
    from asr.utils.gpw2eigs import gpw2eigs
    import numpy as np
    from ase.dft.kpoints import kpoint_convert
    from ase.units import Bohr, Hartree
    from asr.magnetic_anisotropy import get_spin_axis
    calc = GPAW(gpw, txt=None)
    ndim = calc.atoms.pbc.sum()

    theta, phi = get_spin_axis()
    e_skn, efermi2 = gpw2eigs(gpw, soc=soc, theta=theta, phi=phi)
    if efermi is None:
        efermi = efermi2
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    vb_ind, cb_ind = get_vb_cb_indices(e_skn=e_skn, efermi=efermi, delta=delta)

    indices = vb_ind if bandtype == 'vb' else cb_ind
    atoms = calc.get_atoms()
    cell_cv = atoms.get_cell()
    ibz_kc = calc.get_ibz_k_points()

    ibz_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ibz_kc)
    masses = {'indices': indices}
    for b in indices:
        e_k = e_skn[b[0], :, b[1]]
        masses[b] = em(kpts_kv=ibz_kv * Bohr,
                       eps_k=e_k / Hartree, bandtype=bandtype, ndim=ndim)

        calc_bs = calculate_bs_along_emass_vecs
        masses[b]['bs_along_emasses'] = calc_bs(masses[b],
                                                soc, bandtype, calc)

    return masses


def calculate_bs_along_emass_vecs(masses_dict, soc,
                                  bt, calc,
                                  erange=500e-3, npoints=91):
    # out = dict(mass_u=masses, eigenvectors_vu=vecs,
    #            ke_v=kmax,
    #            c=c,
    #            r=r)
    from ase.units import Hartree, Bohr
    from ase.dft.kpoints import kpoint_convert
    from asr.utils.gpw2eigs import gpw2eigs
    from asr.utils.symmetry import is_symmetry_protected
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index
    from asr.core import read_json
    import numpy as np
    cell_cv = calc.get_atoms().get_cell()

    results_dicts = []
    for u, mass in enumerate(masses_dict['mass_u']):
        if mass is np.nan or np.isnan(mass) or mass is None:
            continue

        # embzcut stuff
        kmax = np.sqrt(2 * abs(mass) * erange / Hartree)
        assert not np.isnan(kmax)
        kd_v = masses_dict['eigenvectors_vu'][:, u]
        assert not (np.isnan(kd_v)).any()
        k_kv = (np.linspace(-1, 1, npoints) * kmax * kd_v.reshape(3, 1)).T
        k_kv += masses_dict['ke_v']
        k_kv /= Bohr
        assert not (np.isnan(k_kv)).any()
        k_kc = kpoint_convert(cell_cv=cell_cv, ckpts_kv=k_kv)
        assert not (np.isnan(k_kc)).any()
        atoms = calc.get_atoms()
        for i, pb in enumerate(atoms.pbc):
            if not pb:
                k_kc[:, i] = 0
        assert not (np.isnan(k_kc)).any()
        calc.set(kpts=k_kc, symmetry='off', txt='temp.txt')
        atoms.get_potential_energy()
        name = 'temp.gpw'
        calc.write(name)

        # Start of collect.py stuff

        theta, phi = get_spin_axis()
        e_km, _, s_kvm = gpw2eigs(name, soc=soc, return_spin=True,
                                  theta=theta, phi=phi)

        sz_km = s_kvm[:, get_spin_index(), :]
        from gpaw.symmetry import atoms2symmetry
        op_scc = atoms2symmetry(calc.get_atoms()).op_scc

        magstate = read_json('results-asr.structureinfo.json')['magstate']
        for idx, kpt in enumerate(calc.get_ibz_k_points()):
            if (magstate == 'NM' and is_symmetry_protected(kpt, op_scc) or
                magstate == 'AFM'):
                sz_km[idx, :] = 0.0

        results_dicts.append(dict(kpts_kc=k_kc,
                                  kpts_kv=k_kv,
                                  e_dft_km=e_km,
                                  sz_dft_km=sz_km,
                                  cell_cv=cell_cv))

    return results_dicts


def get_vb_cb_indices(e_skn, efermi, delta):
    """
    find CB and VB within a distance of delta of the CB and VB extrema
    Parameters:
        e_skn: (ns, nk, nb)-shape ndarray
            eigenvalues
        efermi: float
            fermi level
        delta: float
            bands within delta of the extrema are included
    Returns:
        vb_indices, cb_indices: [(spin, band), ..], [(spin, band), ...]
            spin and band indices (aka as SBandex) for VB and CB, respectively
    """
    import numpy as np
    from ase.dft.bandgap import bandgap
    if e_skn.ndim == 2:
        e_skn = e_skn[np.newaxis]
    gap, (s1, k1, n1), (s2, k2, n2) = bandgap(eigenvalues=e_skn,
                                              efermi=efermi, output=None)

    if not gap > 0:
        raise ValueError('Band gap is zero')

    cbm = e_skn[s2, k2, n2]
    vbm = e_skn[s1, k1, n1]

    cb_sn = e_skn[:, k2, n2:]
    vb_sn = e_skn[:, k1, :n1 + 1]
    cbs, cbn = np.where(cb_sn <= cbm + delta)
    cbn += n2
    cb_indices = list(zip(cbs, cbn))

    vbs, vbn = np.where(vb_sn >= vbm - delta)
    vb_indices = list(reversed(list(zip(vbs, vbn))))
    return vb_indices, cb_indices


def em(kpts_kv, eps_k, bandtype=None, ndim=3):
    """
    Parameters:
        kpts_kv: (nk, 3)-shape ndarray
            k-points in cartesian coordinates (in units of 1 / Bohr)
        eps_k: (nk,)-shape ndarray
            eigenvalues (in units of Hartree)
    Returns:
        out: dct
            - effective masses in units of m_e
            - eigenvectors in cartesian coordinates
            - k-pot extremum in cartesian coordinates (units of 1 / Bohr)

    """
    import numpy as np
    c, r, rank, s, = fit(kpts_kv, eps_k, thirdorder=False)
    dxx = 2 * c[0]
    dyy = 2 * c[1]
    dzz = 2 * c[2]
    dxy = c[3]
    dxz = c[4]
    dyz = c[5]

    xm, ym, zm = get_2nd_order_extremum(c, ndim=ndim)
    ke2_v = np.array([xm, ym, zm])

    c3, r3, rank3, s3 = fit(kpts_kv, eps_k, thirdorder=True)

    f3xx, f3yy, f3zz, f3xy = c3[:4]
    f3xz, f3yz, f3x, f3y = c3[4:8]
    f3z, f30, f3xxx, f3yyy = c3[8:12]
    f3zzz, f3xxy, f3xxz, f3yyx, f3yyz, f3zzx, f3zzy, f3xyz = c3[12:]

    if ndim == 2:
        def check_zero(v, i):
            assert np.allclose(v, 0), "Value was {} for index {}".format(v, i)

        zeros = [f3zz, f3xz, f3yz, f3z, f3zzz, f3xxz,
                 f3yyz, f3zzx, f3zzy, f3xyz]
        for i, z in enumerate(zeros):
            check_zero(z, i)

    extremum_type = get_extremum_type(dxx, dyy, dzz, dxy, dxz, dyz, ndim=ndim)
    # if bandtype == 'vb':
    #     assert extremum_type == 'max'
    # elif bandtype == 'cb':
    #     assert extremum_type == 'min'
    # else:
    #     raise NotImplementedError("Incorrect bandtype: {}".format(bandtype))
    xm, ym, zm = get_3rd_order_extremum(xm, ym, zm, c3,
                                        extremum_type, ndim=ndim)
    ke_v = np.array([xm, ym, zm])

    assert not (np.isnan(ke_v)).any()
    if ndim == 2:
        assert np.allclose(zm, 0), zm

    d3xx = (2 * f3xx) + (6 * f3xxx * xm) + (2 * f3xxy * ym) + (2 * f3xxz * zm)
    d3yy = (2 * f3yy) + (6 * f3yyy * ym) + (2 * f3yyx * xm) + (2 * f3yyz * zm)
    d3zz = (2 * f3zz) + (6 * f3zzz * zm) + (2 * f3zzx * xm) + (2 * f3zzy * ym)
    d3xy = f3xy + (2 * f3xxy * xm) + (2 * f3yyx * ym) + (f3xyz * zm)
    d3xz = f3xz + (2 * f3xxz * xm) + (2 * f3zzx * zm) + (f3xyz * ym)
    d3yz = f3yz + (2 * f3yyz * ym) + (2 * f3zzy * zm) + (f3xyz * xm)

    hessian3 = np.array([[d3xx, d3xy, d3xz],
                         [d3xy, d3yy, d3yz],
                         [d3xz, d3yz, d3zz]])

    v3_n, w3_vn = np.linalg.eigh(hessian3)
    assert not (np.isnan(w3_vn)).any()

    hessian = np.array([[dxx, dxy, dxz],
                        [dxy, dyy, dyz],
                        [dxz, dyz, dzz]])
    v2_n, vecs = np.linalg.eigh(hessian)

    mass_u = 1 / v3_n
    for u, v3 in enumerate(v3_n):
        if np.allclose(v3, 0):
            mass_u[u] = np.nan

    sort_args = np.argsort(mass_u)

    mass_u[sort_args]
    for u, m in enumerate(mass_u):
        if np.isnan(m):
            mass_u[u] = None

    w3_vn[sort_args, :]

    out = dict(mass_u=mass_u,
               eigenvectors_vu=w3_vn,
               ke_v=ke_v,
               c=c3,
               r=r3,
               mass2_u=1 / v2_n,
               eigenvectors2_vu=vecs,
               ke2_v=ke2_v,
               c2=c,
               r2=r,
               fitkpts_kv=kpts_kv,
               fite_k=eps_k)

    return out


def get_extremum_type(dxx, dyy, dzz, dxy, dxz, dyz, ndim=3):
    # Input: 2nd order derivatives at the extremum point
    import numpy as np
    if ndim == 3:
        hessian = np.array([[dxx, dxy, dxz],
                            [dxy, dyy, dyz],
                            [dxz, dyz, dzz]])
        vals, vecs = np.linalg.eigh(hessian)
        saddlepoint = not (np.sign(vals[0]) == np.sign(vals[1])
                           and np.sign(vals[0]) == np.sign(vals[2]))

        if saddlepoint:
            etype = 'saddlepoint'
        elif (vals < 0).all():
            etype = 'max'
        elif (vals > 0).all():
            etype = 'min'
        else:
            msg = 'Extremum type could not be' \
                  + 'found for hessian: {}'.format(hessian)
            raise ValueError(msg)
        return etype
    elif ndim == 2:
        # Assume x and y axis are the periodic directions
        hessian = np.array([[dxx, dxy],
                            [dxy, dyy]])
        det = np.linalg.det(hessian)
        if det < 0:
            etype = 'saddlepoint'
        elif dxx < 0 and dyy < 0:
            etype = 'max'
        elif dxx > 0 and dyy > 0:
            etype = 'min'
        else:
            raise ValueError('Extremum type could not'
                             + 'be determined for hessian: {}'.format(hessian))
        return etype
    elif ndim == 1:
        # Assume z axis is the periodic direction
        if dzz < 0:
            etype = 'max'
        else:
            etype = 'min'
        return etype


def get_2nd_order_extremum(c, ndim=3):
    import numpy as np
    # fit is
    # fxx x^2 + fyy y^2 + fzz z^2 +
    # fxy xy + fxz xz + fyz yz + fx x + fy y + fz z + f0
    assert len(c) == 10
    fxx, fyy, fzz, fxy, fxz, fyz, fx, fy, fz, f0 = c

    if ndim == 3:
        ma = np.array([[2 * fxx, fxy, fxz],
                       [fxy, 2 * fyy, fyz],
                       [fxz, fyz, 2 * fzz]])

        v = np.array([-fx, -fy, -fz])

        min_pos = np.linalg.solve(ma, v)

        return min_pos

    elif ndim == 2:
        # Assume x and y are periodic directions
        ma = np.array([[2 * fxx, fxy],
                       [fxy, 2 * fyy]])
        v = np.array([-fx, -fy])
        min_pos = np.linalg.solve(ma, v)

        return np.array([min_pos[0], min_pos[1], 0.0])

    elif ndim == 1:
        # Assume z is periodic direction
        return np.array([0.0, 0.0, -fz / (2 * fzz)])


def get_3rd_order_extremum(xm, ym, zm, c, extremum_type, ndim=3):
    import numpy as np
    from scipy import optimize

    assert len(c) == 20

    if ndim == 3:
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            return model(k)
    elif ndim == 2:
        # Assume x and y are periodic
        # Remove z-dependence
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            m = model(k)
            return m
    elif ndim == 1:
        # Assume z is periodic
        # Remove x, y - dependence
        def get_v(kpts):
            k = np.asarray(kpts)
            if k.ndim == 1:
                k = k[np.newaxis]
            m = model(k)
            return m

    # We want to use a minimization function from scipy
    # so if the extremum type is a 'max' we need to multiply
    # the function by - 1
    if extremum_type == 'max':
        def func(v):
            return -1 * np.dot(get_v(v), c)
    else:
        def func(v):
            return np.dot(get_v(v), c)

    x0 = np.array([xm, ym, zm])
    x, y, z = optimize.fmin(func, x0=x0,
                            xtol=1.0e-15, ftol=1.0e-15, disp=False)

    if ndim == 2:
        return x, y, 0
    elif ndim == 1:
        return 0, 0, z
    else:
        return x, y, z


def fit(kpts_kv, eps_k, thirdorder=False):
    import numpy.linalg as la
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :10]
    return la.lstsq(A_kp, eps_k, rcond=-1)


def model(kpts_kv):
    """ simple third order model
        Parameters:
            kpts_kv: (nk, 3)-shape ndarray
                units of (1 / Bohr)
    """
    import numpy as np
    k_kx, k_ky, k_kz = kpts_kv[:, 0], kpts_kv[:, 1], kpts_kv[:, 2]

    ones = np.ones(len(k_kx))

    A_dp = np.array([k_kx**2,
                     k_ky**2,
                     k_kz**2,
                     k_kx * k_ky,
                     k_kx * k_kz,
                     k_ky * k_kz,
                     k_kx,
                     k_ky,
                     k_kz,
                     ones,
                     k_kx**3,
                     k_ky**3,
                     k_kz**3,
                     k_kx**2 * k_ky,
                     k_kx**2 * k_kz,
                     k_ky**2 * k_kx,
                     k_ky**2 * k_kz,
                     k_kz**2 * k_kx,
                     k_kz**2 * k_ky,
                     k_kx * k_ky * k_kz]).T

    return A_dp


def fit_2d(kpts_kv, eps_k, thirdorder=False):
    import numpy.linalg as la
    A_kp = model_2d(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :6]
    c, r, rank, s = la.lstsq(A_kp, eps_k, rcond=-1)
    return c, r, rank, s


def model_2d(kpts_kv):
    """ simple third order model
        Parameters:
            kpts_kv: (nk, 3)-shape ndarray
                units of (1 / Bohr)
    """
    import numpy as np
    k_kx, k_ky = kpts_kv[:, 0], kpts_kv[:, 1]

    ones = np.ones(len(k_kx))

    A_dp = np.array([k_kx**2,
                     k_ky**2,
                     k_kx * k_ky,
                     k_kx,
                     k_ky,
                     ones,
                     k_kx**3,
                     k_ky**3,
                     k_kx**2 * k_ky,
                     k_ky**2 * k_kx]).T

    return A_dp


def _savemass(soc, bt, mass):
    from ase.parallel import world
    import numpy as np
    fname = get_name(soc, bt) + '.npz'
    if world.rank == 0:
        mass2 = {}
        for k, v in mass.items():
            if type(k) == tuple:
                mass2[k] = v
            elif k == 'indices':
                mass2[k] = [tuple(vi) for vi in v]
            else:
                mass2[k] = v
        with open(fname, 'wb') as f:
            np.savez(f, data=mass2)
    world.barrier()


def _readmass(soc, bt):
    import numpy as np
    fname = get_name(soc=soc, bt=bt) + '.npz'
    with open(fname, 'rb') as f:
        dct = dict(np.load(f))['data'].tolist()
    return dct


def collect_data(atoms):
    from pathlib import Path
    all_data = {}
    kvp = {}
    key_descriptions = {}
    if not list(Path('.').glob('em_circle_*.npz')):
        return {}, {}, {}

    for soc in [True, False]:
        keyname = 'soc' if soc else 'nosoc'
        data = {}
        for bt in ['cb', 'vb']:
            temp = _readmass(soc, bt)
            for key, val in temp.items():
                if key == 'indices':
                    continue
                else:
                    data[bt] = val
        all_data[keyname] = data

    descs = [('Conduction Band emasses',
              'Effective masses for conduction band', '-'),
             ('Valence Band emasses',
              'Effective masses for conduction band', '-'),
             ('Conduction Band emasses with SOC',
              'Effective masses with spin-orbit coupling for conduction band',
              '-'),
             ('Valence Band emasses with SOC',
              'Effective masses with spin-orbit coupling for valence band',
              '-')
             ]

    for socname, socdata in all_data.items():
        soc = socname == 'soc'

        def namemod(n):
            return n + '_soc' if soc else n
        for bt, btdata in socdata.items():
            key = bt + '_emass'
            key = namemod(key)
            kvp[key] = btdata

            if soc:
                key_descriptions[key] = descs[0] if bt == 'cb' else descs[1]
            else:
                key_descriptions[key] = descs[2] if bt == 'cb' else descs[3]

    return kvp, key_descriptions, all_data


def evalmodel(kpts_kv, c_p):
    import numpy as np
    kpts_kv = np.asarray(kpts_kv)
    if kpts_kv.ndim == 1:
        kpts_kv = kpts_kv[np.newaxis]
    A_kp = model(kpts_kv)
    return np.dot(A_kp, c_p)


def evalmodel_2d(kpts_kv, c_p):
    import numpy as np
    kpts_kv = np.asarray(kpts_kv)
    if kpts_kv.ndim == 1:
        kpts_kv = kpts_kv[np.newaxis]
    A_kp = model_2d(kpts_kv)
    num_ks = kpts_kv.shape[0]
    assert A_kp.shape == (num_ks, 10)
    return np.dot(A_kp, c_p)


# def webpanel(row, key_descriptions):

#     from asr.browser import fig
#     add_nosoc = ['D_vbm', 'D_cbm', 'is_metallic', 'is_dir_gap',
#                  'emass1', 'emass2', 'hmass1', 'hmass2', 'work_function']

#     def nosoc_update(string):
#         if string.endswith(')'):
#             return string[:-1] + ', no SOC)'
#         else:
#             return string + ' (no SOC)'

#     for key in add_nosoc:
#         s, l, units = key_descriptions[key]
#         if l:
#             key_descriptions[key + "_nosoc"] = (s, nosoc_update(l), units)
#         else:
#             key_descriptions[key + "_nosoc"] = (nosoc_update(s), l, units)

#     panel = ('Effective masses (PBE)',
#              [[fig('pbe-bzcut-cb-bs.png'), fig('pbe-bzcut-vb-bs.png')],
#               emtables(row)])

#     return panel


if __name__ == '__main__':
    main.cli()
