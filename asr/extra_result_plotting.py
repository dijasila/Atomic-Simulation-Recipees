import numpy as np
from asr.utils.hacks import gs_xcname_from_row
from ase.dft.kpoints import labels_from_kpts


# Infrared: ludicrous-nested plot calls by create_plot from infrared
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


# bandstructure: extra plot
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


### BSE: extra plot function
def gaps_from_row(row):
    for method in ['_gw', '_hse', '_gllbsc', '']:
        gapkey = f'gap_dir{method}'
        if gapkey in row:
            gap_dir_x = row[gapkey]
            delta_bse = gap_dir_x - row.gap_dir
            delta_rpa = gap_dir_x - row.gap_dir_nosoc
            return delta_bse, delta_rpa
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


# charge_neutrality: extra plot functions
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


# chc
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
# convex_hull
from ase.phasediagram import PhaseDiagram
from matplotlib import patches
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


# defect symmetry
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
# defect info


