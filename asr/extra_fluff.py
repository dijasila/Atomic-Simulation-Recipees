from collections import defaultdict

import numpy as np
from matplotlib import patches

from ase.dft.kpoints import labels_from_kpts
from ase.phasediagram import PhaseDiagram
from asr.database.browser import bold, br, code, describe_entry, div, dl, \
    entry_parameter_description, href, \
    table
from asr.result.randomresults import Level
from asr.result.resultdata import model
from asr.utils.hacks import gs_xcname_from_row


def create_plot_simple(*, ndim, omega_w, fname, maxomega, alpha_w,
                       alphavv_w, axisname,
                       omegatmp_w):
    from scipy.interpolate import interp1d

    re_alpha = interp1d(omegatmp_w, alpha_w.real)
    im_alpha = interp1d(omegatmp_w, alpha_w.imag)
    a_w = (re_alpha(omega_w) + 1j * im_alpha(omega_w) + alphavv_w)

    if ndim == 3:
        ylabel = r'Dielectric function'
        yvalues = 1 + 4 * np.pi * a_w
    else:
        power_txt = {2: '', 1: '^2', 0: '^3'}[ndim]
        unit = rf"$\mathrm{{\AA}}{power_txt}$"
        ylabel = rf'Polarizability [{unit}]'
        yvalues = a_w

    return mkplot(yvalues, axisname, fname, maxomega, omega_w, ylabel)


def mkplot(a_w, axisname, fname, maxomega, omega_w, ylabel):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(omega_w, a_w.real, c='C1', label='real')
    ax.plot(omega_w, a_w.imag, c='C0', label='imag')
    ax.set_title(f'Polarization: {axisname}')
    ax.set_xlabel('Energy [meV]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, maxomega)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    return fname


def add_bs_ks(row, ax, reference=0, color='C1'):
    """Plot with soc on ax."""
    d = row.data.get('results-asr.bandstructure.json')
    path = d['bs_soc']['path']
    e_mk = d['bs_soc']['energies']
    xcname = gs_xcname_from_row(row)
    xcoords, label_xcoords, labels = labels_from_kpts(path.kpts, row.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k - reference, color=color, zorder=-2)
    ax.lines[-1].set_label(xcname)
    ef = d['bs_soc']['efermi']
    ax.axhline(ef - reference, ls=':', zorder=-2, color=color)
    return ax


def plot_with_colors(bs, ax=None, filename=None, show=None, energies=None,
                     colors=None, colorbar=True, clabel='$s_z$', cmin=-1.0,
                     cmax=1.0, sortcolors=False, loc=None, s=2):
    """Plot band-structure with colors."""
    import matplotlib.pyplot as plt

    def vlines2back(lines):
        zmin = min([l.get_zorder() for l in lines])
        for l in lines:
            x = l.get_xdata()
            if len(x) > 0 and np.allclose(x, x[0]):
                l.set_zorder(zmin - 1)

    vlines2back(ax.lines)
    shape = energies.shape
    xcoords = np.vstack([bs.xcoords] * shape[0])
    if sortcolors:
        perm = (-colors).argsort(axis=None)
        energies = energies.ravel()[perm].reshape(shape)
        colors = colors.ravel()[perm].reshape(shape)
        xcoords = xcoords.ravel()[perm].reshape(shape)

    for e_k, c_k, x_k in zip(energies, colors, xcoords):
        things = ax.scatter(x_k, e_k, c=c_k, s=s, vmin=cmin, vmax=cmax)

    if colorbar:
        cbar = plt.colorbar(things)
        cbar.set_label(clabel)
    else:
        cbar = None

    bs.finish_plot(filename, show, loc)

    return ax, cbar


def legend_on_top(ax, **kwargs):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1, 1, 0),
              mode='expand', **kwargs)


def _absorption(*, dim, magstate, gap_dir, gap_dir_nosoc,
                bse_data, pol_data,
                delta_bse, delta_rpa, filename, direction):
    import matplotlib.pyplot as plt
    from ase.units import alpha, Ha, Bohr

    qp_gap = gap_dir + delta_bse

    if magstate != 'NM':
        qp_gap = gap_dir_nosoc + delta_rpa
        delta_bse = delta_rpa

    ax = plt.figure().add_subplot(111)

    wbse_w = bse_data[:, 0] + delta_bse
    if dim == 2:
        sigma_w = -1j * 4 * np.pi * (bse_data[:, 1] + 1j * bse_data[:, 2])
        sigma_w *= wbse_w * alpha / Ha / Bohr
        absbse_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
    else:
        absbse_w = 4 * np.pi * bse_data[:, 2]
    ax.plot(wbse_w, absbse_w, '-', c='0.0', label='BSE')
    xmax = wbse_w[-1]

    # TODO: Sometimes RPA pol doesn't exist, what to do?
    if pol_data:
        wrpa_w = pol_data['frequencies'] + delta_rpa
        wrpa_w = pol_data['frequencies'] + delta_rpa
        if dim == 2:
            sigma_w = -1j * 4 * np.pi * pol_data[f'alpha{direction}_w']
            sigma_w *= wrpa_w * alpha / Ha / Bohr
            absrpa_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
        else:
            absrpa_w = 4 * np.pi * np.imag(pol_data[f'alpha{direction}_w'])
        ax.plot(wrpa_w, absrpa_w, '-', c='C0', label='RPA')
        ymax = max(np.concatenate([absbse_w[wbse_w < xmax],
                                   absrpa_w[wrpa_w < xmax]])) * 1.05
    else:
        ymax = max(absbse_w[wbse_w < xmax]) * 1.05

    ax.plot([qp_gap, qp_gap], [0, ymax], '--', c='0.5',
            label='Direct QP gap')

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)
    ax.set_title(f'Polarization: {direction}')
    ax.set_xlabel('Energy [eV]')
    if dim == 2:
        ax.set_ylabel('Absorbance [%]')
    else:
        ax.set_ylabel(r'$\varepsilon(\omega)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

    return ax


def set_labels_and_legend(ax, title):
    ax.set_xlabel(r'$E_\mathrm{F} - E_{\mathrm{VBM}}$ [eV]')
    ax.set_ylabel(f'$E^f$ [eV]')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=5, loc='lower center')


def draw_ef(ax, ef):
    ax.axvline(ef, color='red', linestyle='dotted',
               label=r'$E_\mathrm{F}^{\mathrm{sc}}$')


def set_limits(ax, gap):
    ax.set_xlim(0 - gap / 10., gap + gap / 10.)


def get_min_el(array):
    elements = []
    for i in range(len(array)):
        elements.append(array[i, 0])
    for i, el in enumerate(elements):
        if el == min(elements):
            return i


def get_crossing_point(y1, y2, q1, q2):
    """
    Calculate the crossing point between two charge states.

    f1 = y1 + x * q1
    f2 = y2 + x * q2
    x * (q1 - q2) = y2 - y1
    x = (y2 - y1) / (q1 - q2)
    """
    return (y2 - y1) / float(q1 - q2)


def clean_array(array):
    index = get_min_el(array)

    return array[index:, :]


def get_y(x, array, index):
    q = array[index, 1]

    return q * x + array[index, 0]


def get_last_element(array, x_axis, y_axis, gap):
    y_cbms = []
    for i in range(len(array)):
        q = array[i, 1]
        eform = array[i, 0]
        y_cbms.append(q * gap + eform)

    x_axis.append(gap)
    y_axis.append(min(y_cbms))

    return x_axis, y_axis


def get_line_segment(array, index, x_axis, y_axis, gap):
    xs = []
    ys = []
    for i in range(len(array)):
        if i > index:
            y1 = array[index, 0]
            q1 = array[index, 1]
            y2 = array[i, 0]
            q2 = array[i, 1]
            crossing = get_crossing_point(y1, y2, q1, q2)
            xs.append(crossing)
            ys.append(q1 * crossing + y1)
        else:
            crossing = 1000
            xs.append(gap + 10)
            ys.append(crossing)
    min_index = index + 1
    for i, x in enumerate(xs):
        q1 = array[index, 1]
        y1 = array[index, 0]
        if x == min(xs) and x > 0 and x < gap:
            min_index = i
            x_axis.append(xs[min_index])
            y_axis.append(q1 * xs[min_index] + y1)

    return min_index, x_axis, y_axis


def plot_background(ax, array_in, gap):
    for i in range(len(array_in)):
        q = array_in[i, 1]
        eform = array_in[i, 0]
        y0 = eform
        y1 = eform + q * gap
        ax.plot([0, gap], [y0, y1], color='grey',
                alpha=0.2)


def plot_lowest_lying(ax, array_in, ef, gap, name, color):
    array_tmp = array_in.copy()
    array_tmp = clean_array(array_tmp)
    xs = [0]
    ys = [array_tmp[0, 0]]
    index, xs, ys = get_line_segment(array_tmp, 0, xs, ys, gap)
    for i in range(len(array_tmp)):
        if len(array_tmp[:, 0]) <= 1:
            break
        index, xs, ys = get_line_segment(array_tmp, index, xs, ys, gap)
        if index == len(array_tmp):
            break
    xs, ys = get_last_element(array_tmp, xs, ys, gap)
    ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel(r'$E_\mathrm{F}$ [eV]')


def draw_band_edges(ax, gap):
    ax.axvline(0, color='black')
    ax.axvline(gap, color='black')
    ax.axvspan(-100, 0, alpha=0.5, color='grey')
    ax.axvspan(gap, 100, alpha=0.5, color='grey')


def filrefs(refs):
    from asr.fere import formulas_eq
    nrefs = []
    visited = []
    for (form, v) in refs:
        seen = False
        for x in visited:
            if formulas_eq(form, x):
                seen = True
                break

        if seen:
            continue
        visited.append(form)

        vals = list(filter(lambda t: formulas_eq(t[0], form), refs))

        minref = min(vals, key=lambda t: t[1])

        nrefs.append(minref)

    return nrefs


def get_hull_energies(pd: PhaseDiagram):
    hull_energies = []
    for ref in pd.references:
        count = ref[0]
        refenergy = ref[1]
        natoms = ref[3]
        decomp_energy, indices, coefs = pd.decompose(**count)
        ehull = (refenergy - decomp_energy) / natoms
        hull_energies.append(ehull)

    return hull_energies


class ObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = patches.Polygon(
            [
                [x0, y0],
                [x0, y0 + height],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + 1 / 4 * width, y0],
            ],
            closed=True, facecolor='C2',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        patch = patches.Polygon(
            [
                [x0 + width, y0],
                [x0 + 1 / 4 * width, y0],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + width, y0 + height],
            ],
            closed=True, facecolor='C3',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def draw_levels_occupations_labels(ax, spin, spin_data, ecbm, evbm, ef,
                                   gap, levelflag):
    """Loop over all states in the gap and plot the levels.

    This function loops over all states in the gap of a given spin
    channel, and dravs the states with labels. If there are
    degenerate states, it makes use of the degeneracy_counter, i.e. if two
    degenerate states follow after each other, one of them will be drawn
    on the left side (degoffset=0, degeneracy_counter=0), the degeneracy
    counter will be increased by one and the next degenerate state will be
    drawn on the right side (degoffset=1, degeneracy_counter=1). Since we
    only deal with doubly degenerate states here, the degeneracy counter
    will be set to zero again after drawing the second degenerate state.

    For non degenerate states, i.e. deg = 1, all states will be drawn
    in the middle and the counter logic is not needed.
    """
    # initialize degeneracy counter and offset
    degeneracy_counter = 0
    degoffset = 0
    for sym in spin_data:
        energy = sym.energy
        is_inside_gap = evbm < energy < ecbm
        if is_inside_gap:
            spin = int(sym.spin)
            irrep = sym.best
            # only do drawing left and right if levelflag, i.e.
            # if there is a symmetry analysis to evaluate degeneracies
            if levelflag:
                deg = [1, 2]['E' in irrep]
            else:
                deg = 1
                degoffset = 1
            # draw draw state on the left hand side
            if deg == 2 and degeneracy_counter == 0:
                degoffset = 0
                degeneracy_counter = 1
            # draw state on the right hand side, set counter to zero again
            elif deg == 2 and degeneracy_counter == 1:
                degoffset = 1
                degeneracy_counter = 0
            # intitialize and draw the energy level
            lev = Level(energy, ax=ax, spin=spin, deg=deg,
                        off=degoffset)
            lev.draw()
            # add occupation arrow if level is below E_F
            if energy <= ef:
                lev.add_occupation(length=gap / 15.)
            # draw label based on irrep
            if levelflag:
                static = None
            else:
                static = 'A'
            lev.add_label(irrep, static=static)


def draw_band_edge(energy, edge, color, *, offset=2, ax):
    if edge == 'vbm':
        eoffset = energy - offset
        elabel = energy - offset / 2
    elif edge == 'cbm':
        eoffset = energy + offset
        elabel = energy + offset / 2

    ax.plot([0, 1], [energy] * 2, color='black', zorder=1)
    ax.fill_between([0, 1], [energy] * 2, [eoffset] * 2, color='grey', alpha=0.5)
    ax.text(0.5, elabel, edge.upper(), color='w', weight='bold', ha='center',
            va='center', fontsize=12)


def get_range(mass, _erange):
    from ase.units import Ha, Bohr
    return (2 * mass * _erange / Ha) ** 0.5 / Bohr


def plot_fit(axes, mass, reference, cell_cv,
             xk2, kpts_kv, fit_coeffs):
    from ase.units import Ha
    emodel_k = evalmodel(kpts_kv,
                         fit_coeffs,
                         thirdorder=False) * Ha - reference
    axes.plot(xk2, emodel_k, c='r', ls='--')


def plot_band(fig, axes, mass, reference, cell_cv,
              xk2, kpts_kv, e_km, sz_km,
              cbarlabel, xlabel, ylabel, title,
              bandtype,
              adjust_view=True, spin_degenerate=False):
    import matplotlib.pyplot as plt
    shape = e_km.shape
    perm = (-sz_km).argsort(axis=None)
    repeated_xcoords = np.vstack([xk2] * shape[1]).T
    flat_energies = e_km.ravel()[perm]
    flat_xcoords = repeated_xcoords.ravel()[perm]

    if spin_degenerate:
        colors = np.zeros_like(flat_energies)
    else:
        colors = sz_km.ravel()[perm]

    scatterdata = axes.scatter(flat_xcoords, flat_energies,
                               c=colors, vmin=-1, vmax=1)

    if adjust_view:
        erange = 0.05  # 50 meV
        if bandtype == 'cb':
            y1 = np.min(e_km[:, -1]) - erange * 0.25
            y2 = np.min(e_km[:, -1]) + erange * 0.75
        else:
            y1 = np.max(e_km[:, -1]) - erange * 0.75
            y2 = np.max(e_km[:, -1]) + erange * 0.25
        axes.set_ylim(y1, y2)

        my_range = get_range(min(MAXMASS, abs(mass)), erange)
        axes.set_xlim(-my_range, my_range)

        cbar = fig.colorbar(scatterdata, ax=axes)
        cbar.set_label(cbarlabel)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.update_ticks()
    plt.locator_params(axis='x', nbins=3)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    plt.tight_layout()


def get_plot_data(fit_data, reference, cell_cv):
    from ase.units import Bohr
    from ase.dft.kpoints import kpoint_convert, labels_from_kpts
    ks = fit_data['kpts_kc']
    e_km = fit_data['e_km'] - reference
    sz_km = fit_data['spin_km']
    xk, y, y2 = labels_from_kpts(kpts=ks, cell=cell_cv, eps=1)
    xk -= xk[-1] / 2

    kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
    kpts_kv *= Bohr

    return kpts_kv, xk, e_km, sz_km


def add_fermi(row, ax, s=0.25):
    from matplotlib import pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    verts = row.data['results-asr.fermisurface.json']['contours'].copy()
    normalize = colors.Normalize(vmin=-1, vmax=1)
    verts[:, :2] /= (2 * np.pi)
    im = ax.scatter(verts[:, 0], verts[:, 1], c=verts[:, -1],
                    s=s, cmap='viridis', marker=',',
                    norm=normalize, alpha=1, zorder=2)

    sdir = row.get('spin_axis', 'z')
    cbar = plt.colorbar(im, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params()
    cbar.set_label(r'$\langle S_{} \rangle $'.format(sdir))


def plot_bs(row,
            filename,
            *,
            bs_label,
            efermi,
            data,
            vbm,
            cbm):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    figsize = (5.5, 5)
    fontsize = 10

    path = data['bandstructure']['path']

    reference = row.get('evac')
    if reference is None:
        reference = efermi
        label = r'$E - E_\mathrm{F}$ [eV]'
    else:
        label = r'$E - E_\mathrm{vac}$ [eV]'

    emin_offset = efermi if vbm is None else vbm
    emax_offset = efermi if cbm is None else cbm
    emin = emin_offset - 3 - reference
    emax = emax_offset + 3 - reference

    e_mk = data['bandstructure']['e_int_mk'] - reference
    x, X, labels = path.get_linear_kpoint_axis()

    # with soc
    style = dict(
        color='C1',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    for e_m in e_mk:
        ax.plot(x, e_m, **style)
    ax.set_ylim([emin, emax])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel(label)
    ax.set_xticks(X)
    ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels])

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    ax.axhline(efermi - reference, c='C1', ls=':')
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, efermi - reference),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # add KS band structure with soc
    if 'results-asr.bandstructure.json' in row.data:
        ax = add_bs_ks(row, ax, reference=row.get('evac', row.get('efermi')),
                       color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label=bs_label)
    legend_on_top(ax, ncol=2)
    plt.savefig(filename, bbox_inches='tight')


def get_ordered_syl_dict(dct_syl, symbols):
    """Order a dictionary with syl keys.

    Parameters
    ----------
    dct_syl : dict
        Dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))
    symbols : list
        Sort symbols after index in this list

    Returns
    -------
    outdct_syl : OrderedDict
        Sorted dct_syl

    """
    from collections import OrderedDict

    # Setup ssili (spin, symbol index, angular momentum index) key
    def ssili(syl):
        s, y, L = syl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(y)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{s}{si}{li}'

    return OrderedDict(sorted(dct_syl.items(), key=lambda t: ssili(t[0])))


def get_yl_colors(dct_syl):
    """Get the color indices corresponding to each symbol and angular momentum.

    Parameters
    ----------
    dct_syl : OrderedDict
        Ordered dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))

    Returns
    -------
    color_yl : OrderedDict
        Color strings for each symbol and angular momentum

    """
    from collections import OrderedDict

    color_yl = OrderedDict()
    c = 0
    for key in dct_syl:
        # Do not differentiate spin by color
        if int(key[0]) == 0:  # if spin is 0
            color_yl[key[2:]] = 'C{}'.format(c)
            c += 1
            c = c % 10  # only 10 colors available in cycler

    return color_yl


def plot_pdos_soc(*args, **kwargs):
    return plot_pdos(*args, soc=True, **kwargs)


def plot_pdos(row, filename, soc=True,
              figsize=(5.5, 5), lw=1):

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    # Check if pdos data is stored in row
    results = 'results-asr.pdos.json'
    pdos = 'pdos_soc' if soc else 'pdos_nosoc'
    if results in row.data and pdos in row.data[results]:
        data = row.data[results][pdos]
    else:
        return

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects

    # Extract raw data
    symbols = data['symbols']
    pdos_syl = get_ordered_syl_dict(data['pdos_syl'], symbols)
    e_e = data['energies'].copy() - row.get('evac', 0)
    ef = data['efermi']

    # Find energy range to plot in
    if soc:
        emin = row.get('vbm', ef) - 3 - row.get('evac', 0)
        emax = row.get('cbm', ef) + 3 - row.get('evac', 0)
    else:
        nosoc_data = row.data['results-asr.gs.json']['gaps_nosoc']
        vbmnosoc = nosoc_data.get('vbm', ef)
        cbmnosoc = nosoc_data.get('cbm', ef)

        if vbmnosoc is None:
            vbmnosoc = ef

        if cbmnosoc is None:
            cbmnosoc = ef

        emin = vbmnosoc - 3 - row.get('evac', 0)
        emax = cbmnosoc + 3 - row.get('evac', 0)

    # Set up energy range to plot in
    i1, i2 = abs(e_e - emin).argmin(), abs(e_e - emax).argmin()

    # Get color code
    color_yl = get_yl_colors(pdos_syl)

    # Figure out if pdos has been calculated for more than one spin channel
    spinpol = False
    for k in pdos_syl.keys():
        if int(k[0]) == 1:
            spinpol = True
            break

    # Set up plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot pdos
    pdosint_s = defaultdict(float)
    for key in pdos_syl:
        pdos = pdos_syl[key]
        spin, symbol, lstr = key.split(',')
        spin = int(spin)
        sign = 1 if spin == 0 else -1

        # Integrate pdos to find suiting pdos range
        pdosint_s[spin] += np.trapz(y=pdos[i1:i2], x=e_e[i1:i2])

        # Label atomic symbol and angular momentum
        if spin == 0:
            label = '{} ({})'.format(symbol, lstr)
        else:
            label = None

        ax.plot(smooth(pdos) * sign, e_e,
                label=label, color=color_yl[key[2:]])

    ax.axhline(ef - row.get('evac', 0), color='k', ls=':')

    # Set up axis limits
    ax.set_ylim(emin, emax)
    if spinpol:  # Use symmetric limits
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    # Annotate E_F
    xlim = ax.get_xlim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.99
    text = plt.text(x0, ef - row.get('evac', 0),
                    r'$E_\mathrm{F}$',
                    fontsize=rcParams['font.size'] * 1.25,
                    ha='right',
                    va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    ax.set_xlabel('Projected DOS [states / eV]')
    if row.get('evac') is not None:
        ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    else:
        ax.set_ylabel(r'$E$ [eV]')

    # Set up legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def get_yl_ordering(yl_i, symbols):
    """Get standardized yl ordering of keys.

    Parameters
    ----------
    yl_i : list
        see get_orbital_ldos
    symbols : list
        Sort symbols after index in this list

    Returns
    -------
    c_i : list
        ordered index for each i
    """
    # Setup sili (symbol index, angular momentum index) key
    def sili(yl):
        y, L = yl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(y)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{si}{li}'

    i_c = [iyl[0] for iyl in sorted(enumerate(yl_i), key=lambda t: sili(t[1]))]
    return [i_c.index(i) for i in range(len(yl_i))]


def get_bs_sampling(bsp, npoints=40):
    """Sample band structure as evenly as possible.

    Allways include special points.

    Parameters
    ----------
    bsp : obj
        ase.spectrum.band_structure.BandStructurePlot object
    npoints : int
        number of k-points to sample along band structure

    Returns
    -------
    chosenx_x : 1d np.array
        chosen band structure coordinates
    k_x : 1d np.array
        chosen k-point indices
    """
    # Get band structure coordinates and unique labels
    xcoords, label_xcoords, orig_labels = bsp.bs.get_labels()
    label_xcoords = np.unique(label_xcoords)

    # Reserve one point for each special point
    nonspoints = npoints - len(label_xcoords)
    assert nonspoints >= 0
    assert npoints <= len(xcoords)

    # Slice xcoords into seperate subpaths
    xcoords_lx = []
    subpl_l = []
    lastx = 0.
    for labelx in label_xcoords:
        xcoords_x = xcoords[np.logical_and(xcoords >= lastx,
                                           xcoords <= labelx)]
        xcoords_lx.append(xcoords_x)
        subpl_l.append(xcoords_x[-1] - xcoords_x[0])  # Length of subpath
        lastx = labelx

    # Distribute trivial k-points based on length of slices
    pathlength = sum(subpl_l)
    unitlength = pathlength / (nonspoints + 1)
    # Floor npoints and length remainder for each subpath
    subpnp_l, subprl_l = np.divmod(subpl_l, unitlength)
    subpnp_l = subpnp_l.astype(int)
    # Distribute remainders
    points_left = nonspoints - np.sum(subpnp_l)
    subpnp_l[np.argsort(subprl_l)[-points_left:]] += 1

    # Choose points on each sub path
    chosenx_x = []
    for subpnp, xcoords_x in zip(subpnp_l, xcoords_lx):
        # Evenly spaced indices
        x_p = np.unique(np.round(np.linspace(0, len(xcoords_x) - 1,
                                             subpnp + 2)).astype(int))
        chosenx_x += list(xcoords_x[x_p][:-1])  # each subpath includes start
    chosenx_x.append(xcoords[-1])  # Add end of path

    # Get k-indeces
    chosenx_x = np.array(chosenx_x)
    x_y, k_y = np.where(chosenx_x[:, np.newaxis] == xcoords[np.newaxis, :])
    x_x, y_x = np.unique(x_y, return_index=True)
    k_x = k_y[y_x]

    return chosenx_x, k_x


def get_pie_slice(theta0, theta, s=36., res=64):
    """Get a single pie slice marker.

    Parameters
    ----------
    theta0 : float
        angle in which to start slice
    theta : float
        angle that pie slice should cover
    s : float
        marker size
    res : int
        resolution of pie (in points around the circumference)

    Returns
    -------
    pie : matplotlib.pyplot.scatter option dictionary
    """
    assert -np.pi / res <= theta0 and theta0 <= 2. * np.pi + np.pi / res
    assert -np.pi / res <= theta and theta <= 2. * np.pi + np.pi / res

    angles = np.linspace(theta0, theta0 + theta,
                         int(np.ceil(res * theta / (2 * np.pi))))
    x = [0] + np.cos(angles).tolist()
    y = [0] + np.sin(angles).tolist()
    xy = np.column_stack([x, y])
    size = s * np.abs(xy).max() ** 2

    return {'marker': xy, 's': size, 'linewidths': 0.0}


def get_pie_markers(weight_xi, scale_marker=True, s=36., res=64):
    """Get pie markers corresponding to a 2D array of weights.

    Parameters
    ----------
    weight_xi : 2d np.array
    scale_marker : bool
        using sum of weights as scale for markersize
    s, res : see get_pie_slice

    Returns
    -------
    pie_xi : list of lists of mpl option dictionaries
    """
    assert np.all(weight_xi >= 0.)

    pie_xi = []
    for weight_i in weight_xi:
        pie_i = []
        # Normalize by total weight
        totweight = np.sum(weight_i)
        r0 = 0.
        for weight in weight_i:
            # Weight fraction
            r1 = weight / totweight

            # Get slice
            pie = get_pie_slice(2 * np.pi * r0,
                                2 * np.pi * r1, s=s, res=res)
            if scale_marker:
                pie['s'] *= totweight

            pie_i.append(pie)
            r0 += r1
        pie_xi.append(pie_i)

    return pie_xi


def f(x, a, b):
    return a * x + b


def get_spg_href(url):
    return href('SpgLib', url)


def describe_crystaltype_entry(spglib):
    crystal_type = describe_entry(
        'Crystal type',
        "The crystal type is defined as "
        + br
        + div(bold('-'.join([code('stoi'),
                             code('spg no.'),
                             code('occ. wyck. pos.')])), 'well well-sm text-center')
        + 'where'
        + dl(
            [
                [code('stoi'), 'Stoichiometry.'],
                [code('spg no.'), f'The space group calculated with {spglib}.'],
                [code('occ. wyck. pos.'),
                 'Alphabetically sorted list of occupied '
                 f'wyckoff positions determined with {spglib}.'],
            ]
        )
    )

    return crystal_type


def describe_pointgroup_entry(spglib):
    pointgroup = describe_entry(
        'Point group',
        f"Point group determined with {spglib}."
    )

    return pointgroup


def equation():
    i = '<sub>i</sub>'
    j = '<sub>j</sub>'
    z = '<sup>z</sup>'
    return (f'E{i} = '
            f'−1/2 J ∑{j} S{i} S{j} '
            f'− 1/2 B ∑{j} S{i}{z} S{j}{z} '
            f'− A S{i}{z} S{i}{z}')


def get_table_row(kpt, band, data):
    row = []
    for comp in ['xx', 'yy', 'xy']:
        row.append(data[kpt][comp][band])
    return np.asarray(row)


def gaps_from_row(row):
    for method in ['_gw', '_hse', '_gllbsc', '']:
        gapkey = f'gap_dir{method}'
        if gapkey in row:
            gap_dir_x = row[gapkey]
            delta_bse = gap_dir_x - row.gap_dir
            delta_rpa = gap_dir_x - row.gap_dir_nosoc
            return delta_bse, delta_rpa


def get_number_of_rows(res, spin, vbm, cbm):
    counter = 0
    for i in range(len(res)):
        if (int(res[i]['spin']) == spin
           and res[i]['energy'] < cbm
           and res[i]['energy'] > vbm):
            counter += 1

    return counter


def get_matrixtable_array(state_results, vbm, cbm, ef,
                          spin, style):
    Nrows = get_number_of_rows(state_results, spin, vbm, cbm)
    state_array = np.empty((Nrows, 5), dtype='object')
    rowlabels = []
    spins = []
    energies = []
    symlabels = []
    accuracies = []
    loc_ratios = []
    for i, row in enumerate(state_results):
        rowname = f"{int(state_results[i]['state']):.0f}"
        label = str(state_results[i]['best'])
        labelstr = label.lower()
        splitstr = list(labelstr)
        if len(splitstr) == 2:
            labelstr = f'{splitstr[0]}<sub>{splitstr[1]}</sub>'
        if state_results[i]['energy'] < cbm and state_results[i]['energy'] > vbm:
            if int(state_results[i]['spin']) == spin:
                rowlabels.append(rowname)
                spins.append(f"{int(state_results[i]['spin']):.0f}")
                energies.append(f"{state_results[i]['energy']:.2f}")
                if style == 'symmetry':
                    symlabels.append(labelstr)
                    accuracies.append(f"{state_results[i]['error']:.2f}")
                    loc_ratios.append(f"{state_results[i]['loc_ratio']:.2f}")
    state_array = np.empty((Nrows, 5), dtype='object')
    rowlabels.sort(reverse=True)

    for i in range(Nrows):
        state_array[i, 1] = spins[i]
        if style == 'symmetry':
            state_array[i, 0] = symlabels[i]
            state_array[i, 2] = accuracies[i]
            state_array[i, 3] = loc_ratios[i]
        state_array[i, 4] = energies[i]
    state_array = state_array[state_array[:, -1].argsort()]

    return state_array, rowlabels


def get_transition_table(row, e_hls):
    """Create table for HOMO-LUMO transition in both spin channels."""

    transition_table = table(row, 'Kohn—Sham HOMO—LUMO gap', [])
    for i, element in enumerate(e_hls):
        transition_table['rows'].extend(
            [[describe_entry(f'Spin {i}',
                             f'KS HOMO—LUMO gap for spin {i} channel.'),
              f'{element:.2f} eV']])

    return transition_table


def get_spin_data(data, spin):
    """Create symmetry result only containing entries for one spin channel."""
    spin_data = []
    for sym in data.data['symmetries']:
        if int(sym.spin) == spin:
            spin_data.append(sym)

    return spin_data


def get_summary_table(result, row):

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    basictable = table(row, 'Defect properties', [])
    pg_string = result.defect_pointgroup
    pg_strlist = list(pg_string)
    sub = ''.join(pg_strlist[1:])
    pg_string = f'{pg_strlist[0]}<sub>{sub}</sub>'
    pointgroup = describe_pointgroup_entry(spglib)
    basictable['rows'].extend(
        [[pointgroup, pg_string]])

    return basictable


def convert_key_to_tuple(key):
    k = key.replace("(", "").replace(")", "")
    ks = k.split(",")
    ks = [k.strip() for k in ks]
    ks = [int(k) for k in ks]
    return tuple(ks)


def _get_parameter_description(row):
    desc = entry_parameter_description(
        row.data,
        'asr.gs@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    return desc


def get_dimtypes():
    """Create a list of all dimensionality types."""
    from itertools import product
    s = set(product([0, 1], repeat=4))
    s2 = sorted(s, key=lambda x: (sum(x), *[-t for t in x]))[1:]
    string = "0123"
    return ["".join(x for x, y in zip(string, s3) if y) + "D" for s3 in s2]


def get_overview_tables(scresult, result, unitstring):
    ef = scresult.efermi_sc
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
        'p-type / n-type balance',
        'Balance of p-/n-type dopability in percent '
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
    if scresult.n0 > 1e-5:
        n0 = scresult.n0
    else:
        n0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Electron carrier concentration',
                         'Equilibrium electron carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{n0:.1e} {unitstring}']])
    if scresult.p0 > 1e-5:
        p0 = scresult.p0
    else:
        p0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Hole carrier concentration',
                         'Equilibrium hole carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{p0:.1e} {unitstring}']])

    return scf_overview, scf_summary


def get_conc_table(result, element, unitstring):
    from asr.defectlinks import get_defectstring_from_defectinfo

    token = element['defect_name']
    from asr.defect_symmetry import DefectInfo
    defectinfo = DefectInfo(defecttoken=token)
    defectstring = get_defectstring_from_defectinfo(
        defectinfo, charge=0)  # charge is only a dummy parameter here
    # remove the charge string from the defectstring again
    clean_defectstring = defectstring.split('(charge')[0]
    scf_table = table(result, f'Eq. concentrations of '
                              f'{clean_defectstring} [{unitstring}]', [])
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

    return scf_table


def get_formation_table(result, defstr):

    formation_table = table(result, 'Defect formation energy', [])
    for element in result.eform:
        formation_table['rows'].extend(
            [[describe_entry(f'{defstr} (q = {element[1]:1d} @ VBM)',
                             description='Formation energy for charge state q '
                                         'at the valence band maximum [eV].'),
              f'{element[0]:.2f} eV']])

    return formation_table


def ehull_table_rows(row, key_descriptions, ehull_long_description,
                     eform_description):
    ehull_table = table(row, 'Stability', ['ehull', 'hform'],
                        key_descriptions)

    # We have to magically hack a description into the arbitrarily
    # nested "table" *grumble*:
    rows = ehull_table['rows']
    if len(rows) == 2:
        # ehull and/or hform may be missing if we run tests.
        # Dangerous and hacky, as always.
        rows[0][0] = describe_entry(rows[0][0],
                                    ehull_long_description)
        rows[1][0] = describe_entry(rows[1][0],
                                    eform_description)
    return ehull_table


def get_symmetry_tables(state_results, vbm, cbm, row, style):
    state_tables = []
    gsdata = row.data.get('results-asr.gs.json')
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref

    E_hls = []
    for spin in range(2):
        state_array, rowlabels = get_matrixtable_array(
            state_results, vbm, cbm, ef, spin, style)
        if style == 'symmetry':
            delete = [2]
            columnlabels = ['Symmetry',
                            # 'Spin',
                            'Localization ratio',
                            'Energy']
        elif style == 'state':
            delete = [0, 2, 3]
            columnlabels = [  # 'Spin',
                'Energy']

        N_homo = 0
        N_lumo = 0
        for i in range(len(state_array)):
            if float(state_array[i, 4]) > ef:
                N_lumo += 1

        E_homo = vbm
        E_lumo = cbm
        for i in range(len(state_array)):
            if float(state_array[i, 4]) > ef:
                rowlabels[i] = f'LUMO + {N_lumo - 1}'
                N_lumo = N_lumo - 1
                if N_lumo == 0:
                    rowlabels[i] = 'LUMO'
                    E_lumo = float(state_array[i, 4])
            elif float(state_array[i, 4]) <= ef:
                rowlabels[i] = f'HOMO — {N_homo}'
                if N_homo == 0:
                    rowlabels[i] = 'HOMO'
                    E_homo = float(state_array[i, 4])
                N_homo = N_homo + 1
        E_hl = E_lumo - E_homo
        E_hls.append(E_hl)

        state_array = np.delete(state_array, delete, 1)
        headerlabels = [f'Orbitals in spin channel {spin}',
                        *columnlabels]

        rows = []
        state_table = {'type': 'table',
                       'header': headerlabels}
        for i in range(len(state_array)):
            if style == 'symmetry':
                rows.append((rowlabels[i],
                             # state_array[i, 0],
                             state_array[i, 1],
                             describe_entry(state_array[i, 2],
                                            'The localization ratio is defined as the '
                                            'volume of the cell divided by the integral'
                                            ' of the fourth power of the '
                                            'wavefunction.'),
                             f'{state_array[i, 3]} eV'))
            elif style == 'state':
                rows.append((rowlabels[i],
                             # state_array[i, 0],
                             f'{state_array[i, 1]} eV'))

        state_table['rows'] = rows
        state_tables.append(state_table)

    transition_table = get_transition_table(row, E_hls)

    return state_tables, transition_table


def get_concentration_row(conc_res, defect_name, q):
    rowlist = []
    for scresult in conc_res.scresults:
        condition = scresult.condition
        for i, element in enumerate(scresult['defect_concentrations']):
            conc_row = describe_entry(
                f'Eq. concentration ({condition})',
                'Equilibrium concentration at self-consistent Fermi level.')
            if element['defect_name'] == defect_name:
                for altel in element['concentrations']:
                    if altel[1] == int(q):
                        concentration = altel[0]
                        rowlist.append([conc_row,
                                        f'{concentration:.1e} cm<sup>-2</sup>'])

    return rowlist


def _explain_bandgap(row, gap_name):
    parameter_description = _get_parameter_description(row)

    if gap_name == 'gap':
        name = 'Band gap'
        adjective = ''
    elif gap_name == 'gap_dir':
        name = 'Direct band gap'
        adjective = 'direct '
    else:
        raise ValueError(f'Bad gapname {gap_name}')

    txt = (f'The {adjective}electronic single-particle band gap '
           'including spin–orbit effects.')

    description = f'{txt}\n\n{parameter_description}'
    return describe_entry(name, description=description)


def vbm_or_cbm_row(title, quantity_name, reference_explanation, value):
    description = (f'Energy of the {quantity_name} relative to the '
                   f'{reference_explanation}. '
                   'Spin–orbit coupling is included.')
    return [describe_entry(title, description=description), f'{value:.2f} eV']


def evalmodel(kpts_kv, c_p, thirdorder=True):
    import numpy as np
    kpts_kv = np.asarray(kpts_kv)
    if kpts_kv.ndim == 1:
        kpts_kv = kpts_kv[np.newaxis]
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :10]
    return np.dot(A_kp, c_p)


MAXMASS = 10  # More that 90% of masses are less than this