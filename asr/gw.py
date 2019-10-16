from asr.core import command, option, read_json
from click import Choice


@command('asr.gw')  # , dependencies=['asr.gs'])
@option('--gs', help='Ground state on which GW is based')
@option('--kptdensity', help='K-point density')
@option('--ecut', help='Plane wave cutoff')
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG', 'G0W0-collect', 'GWG-collect']))
@option('--verbose', help='verbose')
def main(gs='gs.gpw', kptdensity=5.0, ecut=200.0, mode='G0W0', verbose=False):
    """Calculate GW"""
    import pickle
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    import gpaw.mpi as mpi
    from gpaw.response.g0w0 import G0W0
    from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
    from pathlib import Path
    import numpy as np

    # check that the system is a semiconductor
    calc = GPAW(gs, txt=None)
    pbe_gap, _, _ = bandgap(calc, output=None)
    if pbe_gap < 0.05:
        raise Exception("GW: Only for semiconductors, PBE gap = " +
                        str(pbe_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, " +
                        str(len(atoms)) + " > 4 atoms!")

    # setup k points/parameters
    dim = np.sum(atoms.pbc.tolist())
    if verbose:
        print("GW: dim =", dim)
    if dim == 3:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        if verbose:
            print("kpt", kptdensity, kpts)
        truncation = 'wigner-seitz'
        q0_correction = False
    elif dim == 2:
        def get_kpts_size(atoms, kptdensity):
            """trying to get a reasonable monkhorst size which hits high
            symmetry points
            """
            from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
            size, offset = k2so(atoms=atoms, density=kptdensity)
            size[2] = 1
            for i in range(2):
                if size[i] % 6 != 0:
                    size[i] = 6 * (size[i] // 6 + 1)
            kpts = {'size': size, 'gamma': True}
            return kpts

        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
        truncation = '2D'
        q0_correction = True
    elif dim == 1:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        # TODO remove unnecessary k
        raise NotImplementedError('asr for dim=1 not implemented!')
        truncation = '1D'
        q0_correction = False
    elif dim == 0:
        kpts = {'density': 0.0, 'gamma': True, 'even': True}
        # TODO only Gamma
        raise NotImplementedError('asr for dim=0 not implemented!')
        truncation = '0D'
        q0_correction = False
    lb, ub = max(calc.wfs.nvalence // 2 - 8, 0), calc.wfs.nvalence // 2 + 4
    if mode.startswith('G0W0'):
        gw_file = 'g0w0_results.pckl'
    elif mode.startswith('GWG'):
        gw_file = 'g0w0_results_GW.pckl'
        raise NotImplementedError('GW: asr for GWG not implemented!')

    # calculate or collect
    if not Path('results-asr.gw.json').is_file():
        # calculate
        if not Path(gw_file).is_file():
            if "collect" in mode:
                raise Exception("GW: No calculation in collect mode!")

            # we need energies/wavefunctions on the correct grid
            if not Path('gs_gw.gpw').is_file():
                if verbose:
                    print("GW: Run gs_gw...")
                calc = GPAW(
                    gs,
                    txt='gs_gw.txt',
                    fixdensity=True,
                    kpts=kpts)
                calc.get_potential_energy()
                calc.diagonalize_full_hamiltonian(ecut=ecut)
                calc.write('gs_gw_nowfs.gpw')
                calc.write('gs_gw.gpw', mode='all')
                mpi.world.barrier()
            else:
                if verbose:
                    print("GW: gs_gw already done")

            if verbose:
                print("GW: Run G0W0...")
            calc = G0W0(calc='gs_gw.gpw',
                        bands=(lb, ub),
                        ecut=ecut,
                        ecut_extrapolation=True,
                        truncation=truncation,
                        nblocksmax=True,
                        q0_correction=q0_correction,
                        filename='g0w0',
                        restartfile='g0w0.tmp',
                        savepckl=True)

            calc.calculate()
            if verbose:
                print("GW: Done")
        else:
            if verbose:
                print("GW: G0W0 already done")

        # Interpolation => currently working for 2D, 3D is not - use
        # same as HSE
        if verbose:
            print("GW: Collect data/Interpolate bs...")
        ranks = [0]
        comm = mpi.world.new_communicator(ranks)
        if mpi.world.rank in ranks:
            if not Path('gs_gw_nowfs.gpw').is_file():
                raise Exception("GW: No gs_gw_nowfs.gpw!")
            calc = GPAW('gs_gw_nowfs.gpw', communicator=comm, txt=None)
            bandrange = list(range(lb, ub))
            gw_skn = pickle.load(open(gw_file, 'rb'), encoding='latin1')['qp']
            theta, phi = 0., 0.  # get_spin_direction()
            e_mk = get_soc_eigs(calc, gw_kn=gw_skn, return_spin=False,
                                theta=theta, phi=phi,
                                bands=bandrange)
            perm_mk = e_mk.argsort(axis=0)
            for e_m, perm_m in zip(e_mk.T, perm_mk.T):
                e_m[:] = e_m[perm_m]

            e_skm = e_mk.T[np.newaxis]
            efermi = fermi_level(calc, e_skm,
                                 nelectrons=(2 *
                                             (calc.get_number_of_electrons() -
                                              bandrange[0] * 2)))

            gap, p1, p2 = bandgap(eigenvalues=e_skm, efermi=efermi,
                                  output=None)
            gapd, p1d, p2d = bandgap(eigenvalues=e_skm, efermi=efermi,
                                     direct=True, output=None)

            if gap <= 0:
                raise Exception("GW: No gap found, gap = " + str(gap) + "!")
            if gap <= pbe_gap:
                raise Exception("GW: Gap smaller than PBE, " +
                                str(gap) + " <= " + str(pbe_gap) + "!")

            try:
                if Path('results_gs.json').is_file():
                    evac = read_json('results_gs.json')
                    evac = evac['vacuumlevels']['evacmean']
                else:
                    vh = GPAW('gs.gpw', txt=None).get_electrostatic_potential()
                    evac1, evac2 = vh.mean(axis=0).mean(axis=0)[[0, -1]]
                    evac = (evac1 + evac2) / 2
            except Exception as x:
                raise Exception("GW: Not able to find evac!")
                print(x)

            vbm = e_skm[p1]
            cbm = e_skm[p2]
            data = dict(gap_gw=gap,
                        bandrange=bandrange,
                        dir_gap_gw=gapd,
                        cbm_gw=cbm - evac,
                        vbm_gw=vbm - evac,
                        efermi=efermi - evac,
                        efermi_gw=efermi - evac)

            data['__key_descriptions__'] = {
                'gap_gw': 'KVP: Band gap (GW) [eV]',
                'dir_gap_gw': 'KVP: Direct band gap (GW) [eV]',
                'efermi_gw': 'KVP: Fermi level (GW) [eV]',
                'cbm_gw': 'KVP: CBM vs. vacuum (GW) [eV]',
                'vbm_gw': 'KVP: VBM vs. vacuum (GW) [eV]'}

            try:
                kpts, e_skm, xreal, epsreal_skn = ip_bs(calc, e_skn=e_skm,
                                                        npoints=400)
            except Exception as x:
                print(x)
            else:
                data.update(path=kpts, eps_skn=e_skm, xreal=xreal,
                            epsreal_skn=epsreal_skn)
    else:
        if verbose:
            print("GW: Already done")
        data = read_json('results-asr.gw.json')

    return data


def eigenvalues(calc):
    """
    Parameters:
        calc: Calculator
            GPAW calculator
    Returns:
        e_skn: (ns, nk, nb)-shape array
    """
    import numpy as np
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def fermi_level(calc, eps_skn=None, nelectrons=None):
    """
    Parameters:
        calc: GPAW
            GPAW calculator
        eps_skn: ndarray, shape=(ns, nk, nb), optional
            eigenvalues (taken from calc if None)
        nelectrons: float, optional
            number of electrons (taken from calc if None)
    Returns:
        out: float
            fermi level
    """
    from gpaw.occupations import occupation_numbers
    from ase.units import Ha
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


def ip_bs(calc, e_skn=None, npoints=400):
    """simple wrapper for interpolate_bandlines2
    Returns:
        out: kpts, e_skn, xreal, epsreal_skn
    """
    path = get_special_path(cell=calc.atoms.cell)
    r = interpolate_bandlines2(calc=calc, path=path, e_skn=e_skn,
                               npoints=npoints)
    return r['kpts'], r['e_skn'], r['xreal'], r['epsreal_skn']


def interpolate_bandlines2(calc, path, e_skn=None, npoints=400):
    """Interpolate bandstructure
    Parameters:
        calc: ASE calculator
        path: str
            something like GMKG
        e_skn: (ns, nk, nb) shape ndarray, optional
            if not given it uses eigenvalues from calc
        npoints: int
            numper of point on the path
    Returns:
        out: dict
            with keys e_skn, kpts, x, X
            e_skn: (ns, npoints, nb) shape ndarray
                interpolated eigenvalues,
            kpts:  (npoints, 3) shape ndarray
                kpts on path (in basis of reciprocal vectors)
            x: (npoints, ) shape ndarray
                x axis
            X: (nkspecial, ) shape ndarrary
                position of special points (G, M, K, G) on x axis

    """
    import numpy as np
    from ase.dft.kpoints import bandpath
    from scipy.interpolate import CubicSpline
    # print("GW: interpolate_bandlines2...")
    if e_skn is None:
        e_skn = eigenvalues(calc)
    kpts = calc.get_bz_k_points()
    bz2ibz = calc.get_bz_to_ibz_map()
    cell = calc.atoms.cell
    indices, x = segment_indices_and_x(cell=cell, path_str=path, kpts=kpts)
    # kpoints and positions to interpolate onto
#   kpts2, x2, X2 = bandpath(cell=cell, path=path, npoints=npoints)
    path2 = bandpath(cell=cell, path=path, npoints=npoints)
    kpts2 = path2.kpts
    x2, X2, _ = path2.get_linear_kpoint_axis()
    # remove double points
    for n in range(len(indices) - 1):
        if indices[n][-1] == indices[n + 1][0]:
            del indices[n][-1]
            x[n] = x[n][:-1]
    # flatten lists [[0, 1], [2, 3, 4]] -> [0, 1, 2, 3, 4]
    indices = [a for b in indices for a in b]
    kptsreal_kc = kpts[indices]
    x = [a for b in x for a in b]
    # loop over spin and bands and interpolate
    ns, nk, nb = e_skn.shape
    e2_skn = np.zeros((ns, len(x2), nb), float)
    epsreal_skn = np.zeros((ns, len(x), nb), float)
    for s in range(ns):
        for n in range(nb):
            e_k = e_skn[s, :, n]
            y = [e_k[bz2ibz[i]] for i in indices]
            epsreal_skn[s, :, n] = y
            bc_type = ['not-a-knot', 'not-a-knot']
            for i in [0, -1]:
                if path[i] == 'G':
                    bc_type[i] = [1, 0.0]
            sp = CubicSpline(x=x, y=y, bc_type=bc_type)
            # sp = InterpolatedUnivariateSpline(x, y)
            e2_skn[s, :, n] = sp(x2)
    results = {'kpts': kpts2,    # kpts_kc on bandpath
               'e_skn': e2_skn,  # eigenvalues on bandpath
               'x': x2,          # distance along bandpath
               'X': X2,          # positons of vertices on bandpath
               'xreal': x,       # distance along path (at MonkhorstPack kpts)
               'epsreal_skn': epsreal_skn,  # path eigenvalues at MP kpts
               'kptsreal_kc': kptsreal_kc   # path k-points at MP kpts
               }
    return results


def ontheline(p1, p2, p3s, eps=1.0e-5):
    """
    line = p1 + t * (p2 - p1)
    check whether p3 is on the line (t is between 0 and 1)
    Parameters:
        p1, p2: ndarray
            point defining line p1, p2 and third point p3 we are checking
        p3s: list [ndarray,]
        eps: float
            slack in distance to be considered on the line
    Returns:
        indices: [(int, float), ] * Np
            indices and t's for p3s on the line,
            i.e [(0, 0.1), (4, 0.2), (3, 1.0], sorted accorting to t
    """
    import numpy as np
    from numpy import linalg as la
    nk = len(p3s)
    kpts = np.zeros((nk * 4, 3))
    kpts[:nk] = p3s
    kpts[nk:2 * nk] = p3s - (1, 0, 0)
    kpts[2 * nk:3 * nk] = p3s - (1, 1, 0)
    kpts[3 * nk:4 * nk] = p3s - (0, 1, 0)
    d = p2 - p1  # direction
    d2 = np.dot(d, d)
    its = []
    for i, p3 in enumerate(kpts):
        t = np.dot(d, p3 - p1) / d2
        x = p1 + t * d  # point on the line that minizes distance to p3
        dist = la.norm(x - p3)
        if (0 - eps <= t <= 1 + eps) and (dist < eps):
            its.append((i % nk, t))
    its = sorted(its, key=lambda x: x[1])
    return its


def segment_indices_and_x(cell, path_str, kpts):
    """finds indices of bz k-points that is located on segments of a bandpath
    Parameters:
        cell: ndarray (3, 3)-shape
            unit cell
        path_str: str
            i.e. 'GMKG'
        kpts: ndarray (nk, 3)-shape
    Returns:
        out: ([[int,] * Np,] * Ns, [[float,] * Np, ] * Ns)
            list of indices and list of x
    """
    import numpy as np
    from ase.dft.kpoints import parse_path_string, get_special_points, bandpath
    special = get_special_points(cell)
    # _, _, X = bandpath(path=path_str, cell=cell, npoints=len(path_str)) # was
    # path = bandpath(path=path_str, cell=cell, npoints=len(path_str)) # wrong
    path = bandpath(path=path_str, cell=cell, npoints=400)
    _, X, _ = path.get_linear_kpoint_axis()
    # why did the old code broke here?
    segments_length = np.diff(X)  # length of band segments
    path = parse_path_string(path_str)[0]  # list str, i.e. ['G', 'M', 'K','G']
    segments_points = []
    # make segments [G,M,K,G] -> [(G,M), (M,K), (K.G)]
    for i in range(len(path) - 1):
        kstr1, kstr2 = path[i:i + 2]
        s1, s2 = special[kstr1], special[kstr2]
        segments_points.append((s1, s2))

    # find indices where kpts is on the segments
    segments_indices = []
    segments_xs = []
    for (k1, k2), d, x0 in zip(segments_points, segments_length, X):
        its = ontheline(k1, k2, kpts)
        indices = [i for i, t in its]
        ts = np.asarray([t for i, t in its])
        xs = ts * d  # positions on the line of length d
        segments_xs.append(xs + x0)
        segments_indices.append(indices)

    return segments_indices, segments_xs


def get_special_path(cell):
    import numpy as np
    from ase.dft.kpoints import special_paths
    from ase.geometry import crystal_structure_from_cell as csfc
    special_2d_paths = {'hexagonal': 'GMKG',
                        'orthorhombic': 'GXSYGS',
                        'tetragonal': 'MGXM',
                        'monoclinic': 'GYHCH1XG',
                        # ...
                        }
    dim = np.sum(cell.pbc.tolist())
    if dim == 3:
        return special_paths[csfc(cell)]
    elif dim == 2:
        return special_2d_paths[csfc(cell)]


def bs_xc(row, path, xc, **kwargs):
    """xc: 'gw' or 'hse'
    """
    from c2db.bsfitfig import bsfitfig
    from asr.bandstructure import add_bs_pbe
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    lw = kwargs.get('lw', 1)
    if row.data.get('bs_pbe', {}).get('path') is None:
        return
    if 'bs_' + xc not in row.data:
        return
    ax = bsfitfig(row, xc=xc, lw=lw)
    if ax is None:
        return
    label = kwargs.get('label', '?')
    # trying to make the legend label look nice
    for line1 in ax.lines:
        if line1.get_marker() == 'o':
            break
    line0 = ax.lines[0]
    line1, = ax.plot([], [],
                     '-o',
                     c=line0.get_color(),
                     markerfacecolor=line1.get_markerfacecolor(),
                     markeredgecolor=line1.get_markeredgecolor(),
                     markersize=line1.get_markersize(),
                     lw=line0.get_lw())
    line1.set_label(label)
    if 'bs_pbe' in row.data and 'path' in row.data.bs_pbe:
        ax = add_bs_pbe(row, ax, **kwargs)
    ef = row.get('efermi_{}'.format(xc))
    ax.axhline(ef, c='k', ls=':')
    emin = row.get('vbm_' + xc, ef) - 3
    emax = row.get('cbm_' + xc, ef) + 3
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    ax.set_ylim(emin, emax)
    ax.set_xlabel('$k$-points')
    leg = ax.legend(loc='upper right')
    leg.get_frame().set_alpha(1)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$', xy=(x0, ef), ha='left', va='bottom', fontsize=13)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])
    plt.savefig(path)
    plt.close()


def bs_gw(row, path):
    bs_xc(row, path, xc='gw', label='G$_0$W$_0$')


def webpanel(row, key_descriptions):
    from asr.browser import fig, table

    prop = table(row, 'Property', [
        'gap_gw', 'dir_gap_gw', 'vbm_gw', 'cbm_gw'
    ], key_descriptions)

    panel = ('Electronic band structure (GW)',
             [[fig('gw-bs.png'), prop]])

    things = [(bs_gw, ['gw-bs.png'])]

    return panel, things


if __name__ == '__main__':
    main.cli()
