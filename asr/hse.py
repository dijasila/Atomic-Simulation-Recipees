"""
to do:
- better interpolation scheme?
- find reasonable default values for params
- move stuff to utils
- get evac
- create tests
- move relevant functions to hseinterpolate? or merge into one single recipe?
"""
import json
from pathlib import Path
from asr.utils import command, option, read_json

import time

from gpaw import GPAW
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
import numpy as np
from numpy import linalg as la
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline

from ase.parallel import paropen, parprint
import os
import gpaw.mpi as mpi
from ase.dft.kpoints import (get_monkhorst_pack_size_and_offset,
                             monkhorst_pack_interpolate,
                             bandpath, parse_path_string, labels_from_kpts)
from ase.io import read
from ase.dft.kpoints import get_cellinfo
from contextlib import contextmanager

@command('asr.hse')
@option('--kptdensity', help='K-point density')
@option('--emptybands', help='number of empty bands to include')
def main(kptdensity=12, emptybands=20):
    """Calculate HSE band structure"""
    results = {}
    results['hse_eigenvalues'] = hse(kptdensity=kptdensity, emptybands=emptybands)
    mpi.world.barrier()
    results['hse_eigenvalues_soc'] = hse_spinorbit(results['hse_eigenvalues'])
    return results

def hse(kptdensity, emptybands):

    convbands = int(emptybands / 2)
    if not os.path.isfile('hse.gpw'):
        calc = GPAW('gs.gpw', txt=None)
        atoms = calc.get_atoms()
        pbc = atoms.pbc.tolist()
        ND = np.sum(pbc)
        if ND == 3 or ND == 1:
            kpts = {'density': kptdensity, 'gamma': False, 'even': True}
        elif ND == 2:

            # move to utils? [also in asr.polarizability]
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

        calc.set(nbands=-emptybands,
                 fixdensity=True,
                 kpts=kpts,
                 convergence={'bands': -convbands},
                 txt='hse.txt')
        calc.get_potential_energy()
        calc.write('hse.gpw', 'all')
        calc.write('hse_nowfs.gpw')
    mpi.world.barrier()
    time.sleep(10)  # is this needed?

    calc = GPAW('hse.gpw', txt=None)
    ns = calc.get_number_of_spins()
    nk = len(calc.get_ibz_k_points())
    nb = calc.get_number_of_bands()

    hse_calc = EXX('hse.gpw', xc='HSE06', bands=[0, nb - convbands])
    hse_calc.calculate(restart='hse-restart.json')
    vxc_hse_skn = hse_calc.get_eigenvalue_contributions()

    vxc_pbe_skn = vxc(calc, 'PBE')[:, :, :-convbands]
    e_pbe_skn = np.zeros((ns, nk, nb))
    for s in range(ns):
        for k in range(nk):
            e_pbe_skn[s, k, :] = calc.get_eigenvalues(spin=s, kpt=k)
    e_pbe_skn = e_pbe_skn[:, :, :-convbands]

    e_hse_skn = e_pbe_skn - vxc_pbe_skn + vxc_hse_skn

    dct = {}
    if mpi.world.rank == 0:
        dct = dict(vxc_hse_skn=vxc_hse_skn,
                   e_pbe_skn=e_pbe_skn,
                   vxc_pbe_skn=vxc_pbe_skn,
                   e_hse_skn=e_hse_skn)
    return dct

def hse_spinorbit(dct):
    if not os.path.isfile('hse_nowfs.gpw'):
        return

    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('hse_nowfs.gpw', communicator=comm, txt=None)
        e_skn = dct.get('e_hse_skn')
        dct_soc = {}
        theta, phi = get_spin_direction()
        e_mk, s_kvm = get_soc_eigs(calc, gw_kn=e_skn, return_spin=True,
                                   bands=np.arange(e_skn.shape[2]),
                                   theta=theta, phi=phi)
        dct_soc['e_hse_mk'] = e_mk
        dct_soc['s_hse_mk'] = s_kvm[:, spin_axis(), :].transpose()

        return dct_soc

def bs_interpolate(kptpath, npoints=400, show=False):
    """inpolate the eigenvalues on a monkhorst pack grid to
    a path in the bz between high-symmetry points.
    The soc is added before interpolation.

    Parameters:
        kptpath: a string such as 'GMKG' (optional)
        npoints: num. of desired points along the bandpath
    Returns:
        out: 2 dictionaries results['hse_bandstructure'] and results['hse_bandstructure3']
            results['hse_bandstructure']: dict with keys eps_skn, path, e_mk, s_mk
            results['hse_bandstructure3']: dict with keys eps_skn, path, e_mk, xreal, epsreal_skn

    """
    calc = GPAW('hse_nowfs.gpw', txt=None)
    atoms = calc.atoms
    results_hse = read_json('results_hse.json')
    data = results_hse['hse_eigenvalues']
    e_skn = data['e_hse_skn']
    e_skn.sort(axis=2)
    try:
        data = results_hse['hse_eigenvalues_soc']
        e_mk = data['e_hse_mk']
        s_mk = data['s_hse_mk']
        perm_mk = e_mk.argsort(axis=0)
        for s_m, e_m, perm_m in zip(s_mk.T, e_mk.T, perm_mk.T):
            e_m[:] = e_m[perm_m]
            s_m[:] = s_m[perm_m]
    except IOError:
        e_mk = None

    size, offset = get_monkhorst_pack_size_and_offset(calc.get_bz_k_points())
    bz2ibz = calc.get_bz_to_ibz_map()
    if kptpath == None:
        path = atoms.cell.bandpath(npoints=npoints)
    else:
        path = atoms.cell.bandpath(kptpath, npoints=npoints)
    icell = atoms.get_reciprocal_cell()
    eps = monkhorst_pack_interpolate(path.kpts, e_skn.transpose(1, 0, 2),
                                     icell, bz2ibz, size, offset) # monkhorst_pack_interpolate wants a (npoints, 3) path array
    eps_skn = eps.transpose(1, 0, 2)
    dct = dict(eps_skn=eps_skn, path=path)
    if e_mk is not None:
        eps_soc = monkhorst_pack_interpolate(path.kpts, e_mk.transpose(1, 0),
                                             icell, bz2ibz, size, offset)
        s_soc = monkhorst_pack_interpolate(path.kpts, s_mk.transpose(1, 0),
                                           icell, bz2ibz, size, offset)
        e_mk = eps_soc.transpose(1, 0)
        s_mk = s_soc.transpose(1, 0)
        dct.update(e_mk=e_mk, s_mk=s_mk)

    results = {}
    results['hse_bandstructure'] = dct

    # XXX: do we really need the following??
    # XXX: interpolate_bandstructure does NOT work well for 3D structures
    hse_eigenvalues = results_hse['hse_eigenvalues']
    eps_skn = hse_eigenvalues['e_hse_skn']
    _, _, _, e_skn, _, _ = interpolate_bandstructure(calc, path, e_skn=e_skn)
    dct = dict(eps_skn=e_skn, path=path)
    hse_eigenvalues_soc = results_hse['hse_eigenvalues_soc']
    eps_smk = hse_eigenvalues_soc['e_hse_mk']
    eps_smk = eps_smk[np.newaxis]
    _, _, _, e_skn, xr, yr_skn = interpolate_bandstructure(calc, path, e_skn=eps_smk.transpose(0, 2, 1))
    dct.update(e_mk=e_skn[0].transpose(), xreal=xr, epsreal_skn=yr_skn)
    results['hse_bandstructure3'] = dct

    return results


# move to utils?
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

# move to utils?
def segment_indices_and_x(cell, path, kpts):
    """finds indices of bz k-points that is located on segments of a bandpath
    Parameters:
        cell: a Cell object
        path: a BandPath object
        kpts: ndarray (nk, 3)-shape
    Returns:
        out: ([[int,] * Np,] * Ns, [[float,] * Np, ] * Ns)
            list of indices and list of x
    """
    from ase.dft.kpoints import parse_path_string, bandpath
    special_points = path.special_points
    _, X, _ = labels_from_kpts(path.kpts, cell, eps=1e-5, special_points=special_points)
    segments_length = np.diff(X)  # length of band segments
    list_str_path = parse_path_string(path.path)[0]  # list str, i.e. ['G', 'M', 'K','G']
    segments_points = []
    # make segments [G,M,K,G] -> [(G,M), (M,K), (K.G)]
    for i in range(len(list_str_path) - 1):
        kstr1, kstr2 = list_str_path[i:i + 2]
        s1, s2 = special_points[kstr1], special_points[kstr2]
        segments_points.append((s1, s2))

    # find indices where kpts is on the segments
    segments_indices = []
    segments_xs = []
    for (k1, k2), d, x0 in zip(segments_points, segments_length, X):
        its = ontheline(k1, k2, kpts)
        """
        Warning: the list returned by ontheline may be empty!
        This may happen if there is no BZ kpoint close enough to the bandpath for one segment
        In such a case we should't append anything to segments_xs and segments_indices
        """
        if len(its)!=0:
            indices = [i for i, t in its]
            ts = np.asarray([t for i, t in its])
            xs = ts * d  # positions on the line of length d
            segments_xs.append(xs + x0)
            segments_indices.append(indices)

    return segments_indices, segments_xs

# move to utils?
def interpolate_bandstructure(calc, path, e_skn=None):
    """simple wrapper for interpolate_bandlines2
    Returns:
        out: kpts, x, X, e_skn, xreal, epsreal_skn

    XXX: you don't really need to return kpts, x, X
    """
    r = interpolate_bandlines2(calc=calc, path=path, e_skn=e_skn)
    return r['kpts'], r['x'], r['X'], r['e_skn'], r['xreal'], r['epsreal_skn']

# move to utils?
def interpolate_bandlines2(calc, path, e_skn=None):
    """Interpolate bandstructure
    Parameters:
        calc: ASE calculator
        path: a BandPath object
        e_skn: (ns, nk, nb) shape ndarray, optional
            if not given it uses eigenvalues from calc
    Returns:
        out: dict with keys kpts, e_skn, x, X, xreal, epsreal_skn, kptsreal_kc
            e_skn: (ns, npoints, nb) shape ndarray
                interpolated eigenvalues,
            kpts:  (npoints, 3) shape ndarray
                kpts on path (in basis of reciprocal vectors)
            x: (npoints, ) shape ndarray
                x axis
            X: (nkspecial, ) shape ndarrary
                position of special points (G, M, K, G) on x axis
            xreal: position of real MonkhorstPack kpts on x axis
            epsreal_skn: real eigenvalues at MP kpts, from which the interpolation is constructed
            kptsreal_kc: real MP kpts

    """
    if e_skn is None:
        e_skn = eigenvalues(calc)
    kpts = calc.get_bz_k_points()
    bz2ibz = calc.get_bz_to_ibz_map()
    cell = calc.atoms.cell

    special_points = path.special_points
    x, X, labels = labels_from_kpts(path.kpts, cell)
        
    """
    Now split disconnected segments into separate paths (if there are any)
    Example: 3D fcc path GXWKGLUWLK,UX -> separate GXWKGLUWLK and UX """
    # create a list of indices where a new segment starts
    list_n = []
    list_n.append(0)
    for n in range(len(x)-1):
        # find indices for which x[n]==x[n+1]
        # they are the values of x where bandpath is disconnected
        if x[n]==x[n+1]:
            list_n.append(n+1)
    list_n.append(len(x-1))

    partial_paths = [] # list of disconnected paths
    for i in range(len(list_n)-1):
        n0 = list_n[i]
        n1 = list_n[i+1]
        # select a portion of total kpts between n0 and n1
        partial_kpts = path.kpts[n0:n1,:]
        partial_n=partial_kpts.shape[0]
        # create partial path from partial_kpts
        partial_path = bandpath(path=partial_kpts, cell=cell, npoints=partial_n)
        # note: you have to assign labels to the new partial_path manually!
        _, _, partial_labels = labels_from_kpts(partial_kpts, path.cell, special_points=path.special_points)
        partial_path.path = ''.join(partial_labels)
        partial_path.special_points = special_points
        partial_paths.append(partial_path)

    # get results for all paths and concatenate

    list_kptsreal_kc = []
    list_epsreal_skn = []
    list_xreal = []
    list_e2_skn = []
 
    for partial_path in partial_paths:
        x2, _, _ = labels_from_kpts(partial_path.kpts, cell, special_points=partial_path.special_points)
        indices, x = segment_indices_and_x(cell=cell, path=partial_path, kpts=kpts)
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
                    if partial_path.path[i] == 'G':
                        bc_type[i] = [1, 0.0]
                if len(x) > 1:
                    sp = CubicSpline(x=x, y=y, bc_type=bc_type)
                    #sp = InterpolatedUnivariateSpline(x, y)
                    e2_skn[s, :, n] = sp(x2)
                else:
                    # XXX: what do we do if len(x)<2? For the moment, create array of zeros
                    e2_skn[s, :, n] = np.zeros(len(x2)) 
        list_e2_skn.append(e2_skn)
        list_kptsreal_kc.append(kptsreal_kc)
        list_epsreal_skn.append(epsreal_skn)
        list_xreal.append(np.array(x))
        
    tot_kptsreal_kc = list_kptsreal_kc[0]
    tot_epsreal_skn = list_epsreal_skn[0]
    tot_xreal = list_xreal[0]
    tot_e2_skn = list_e2_skn[0]
      
    for i in range(1, len(partial_paths)):
        tot_kptsreal_kc = np.concatenate((tot_kptsreal_kc, list_kptsreal_kc[i]), axis=0)
        tot_epsreal_skn = np.concatenate((tot_epsreal_skn, list_epsreal_skn[i]), axis=1)
        tot_xreal = np.concatenate((tot_xreal, list_xreal[i] + tot_xreal[-1]))
        tot_e2_skn = np.concatenate((tot_e2_skn, list_e2_skn[i]), axis=1)

    tot_kpts = path.kpts
    tot_x, tot_X, _ = labels_from_kpts(tot_kpts, cell)
    results = {'kpts': tot_kpts,    # kpts_kc on bandpath
               'e_skn': tot_e2_skn,  # eigenvalues on bandpath
               'x': tot_x,     # distance along bandpath
               'X': tot_X,     # positons of vertices on bandpath
               'xreal': tot_xreal,       # distance along path (at MonkhorstPack kpts)
               'epsreal_skn': tot_epsreal_skn,  # path eigenvalues at MP kpts
               'kptsreal_kc': tot_kptsreal_kc   # path k-points at MP kpts
               }
    return results

# move to utils? [also in asr.bandstructure]
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

# move to utils? [also in asr.bandstructure]
def get_spin_direction(fname='anisotropy_xy.npz'):
    '''
    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Parameters:
        fname:
            The filename of a datafile containing the xz and yz
            anisotropy energies.
    Returns:
        theta:
            Polar angle in radians
        phi:
            Azimuthal angle in radians
    '''

    import numpy as np
    import os
    theta = 0
    phi = 0
    if os.path.isfile(fname):
        data = np.load(fname)
        DE = max(data['dE_zx'], data['dE_zy'])
        if DE > 0:
            theta = np.pi / 2
            if data['dE_zy'] > data['dE_zx']:
                phi = np.pi / 2
    return theta, phi

# move to utils? [also in asr.bandstructure]
def spin_axis(fname='anisotropy_xy.npz') -> int:
    import numpy as np
    theta, phi = get_spin_direction(fname=fname)
    if theta == 0:
        return 2
    elif np.allclose(phi, np.pi / 2):
        return 1
    else:
        return 0

# move to utils?
@contextmanager
def cleanup(*files):
    try:
        yield
    finally:
        mpi.world.barrier()
        if mpi.world.rank == 0:
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)

def collect_data(atoms):
    from ase.dft.bandgap import bandgap
    kvp = {}
    key_descriptions = {}
    data = {}

    evac = 0.0 # XXX where do I find evac?
    #evac = kvp.get('evac')

    if not os.path.isfile('results-asr.hse.json'):
        return kvp, key_descriptions, data

    results_hse = read_json('results-asr.hse.json')
    
    eps_skn = results_hse['hse_eigenvalues']['e_hse_skn']
    calc = GPAW('hse_nowfs.gpw', txt=None)
    ibzkpts = calc.get_ibz_k_points()


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
        if nelectrons is None:
            nelectrons = calc.get_number_of_electrons()
        if eps_skn is None:
            eps_skn = eigenvalues(calc)
        eps_skn.sort(axis=-1)
        occ = calc.occupations.todict()
        weight_k = calc.get_k_point_weights()
        from gpaw.occupations import occupation_numbers
        from ase.units import Ha
        return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


    efermi_nosoc = fermi_level(calc, eps_skn=eps_skn)
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if 'hse' not in data:
        data['hse'] = {}
    if gap:
        data['hse']['kvbm_nosoc'] = ibzkpts[p1[1]] # k coordinates of vbm
        data['hse']['kcbm_nosoc'] = ibzkpts[p2[1]] # k coordinates of cbm
        vbm = eps_skn[p1] - evac
        cbm = eps_skn[p2] - evac
        kvp.update(vbm_hse_nosoc=vbm, cbm_hse_nosoc=cbm,
                   dir_gap_hse_nosoc=gapd, gap_hse_nosoc=gap)

    eps = results_hse['hse_eigenvalues_soc']['e_hse_mk']
    eps = eps.transpose()[np.newaxis]  # e_skm, dummy spin index
    efermi_soc = fermi_level(calc, eps_skn=eps,
                         nelectrons=calc.get_number_of_electrons() * 2)
    gap, p1, p2 = bandgap(eigenvalues=eps, efermi=efermi_soc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps, efermi=efermi_soc,
                             direct=True, output=None)
    if gap:
        data['hse']['kvbm'] = ibzkpts[p1[1]]
        data['hse']['kcbm'] = ibzkpts[p2[1]]
        vbm = eps[p1] - evac
        cbm = eps[p2] - evac
        kvp.update(vbm_hse=vbm, cbm_hse=cbm,
                   dir_gap_hse=gapd, gap_hse=gap)
    kvp.update(efermi_hse_nosoc=efermi_nosoc - evac,
               efermi_hse_soc=efermi_soc - evac)
      
    kd = {
        'vbm_hse_nosoc': ('HSE Valence Band Max - no soc',
                          'HSE Valence Band Maximum without spin-orbit coupling',
                          'eV'),
        'cbm_hse_nosoc': ('HSE Conduction Band Min - no soc',
                          'HSE Conduction Band Minimum without spin-orbit coupling',
                          'eV'),
        'dir_gap_hse_nosoc': ('HSE direct gap - no soc',
                              'HSE direct gap without spin-orbit coupling',
                              'eV'),
        'gap_hse_nosoc': ('HSE gap - no soc',
                          'HSE gap without spin-orbit coupling',
                          'eV'),
        'vbm_hse': ('HSE Valence Band Max - soc',
                    'HSE Valence Band Maximum with spin-orbit coupling',
                    'eV'),
        'cbm_hse': ('HSE Conduction Band Min - soc',
                    'HSE Conduction Band Minimum with spin-orbit coupling',
                    'eV'),
        'dir_gap_hse': ('HSE direct gap - soc',
                        'HSE direct gap with spin-orbit coupling',
                        'eV'),
        'gap_hse': ('HSE gap - soc',
                    'HSE gap with spin-orbit coupling',
                    'eV'),
        'efermi_hse_nosoc': ('HSE Fermi energy - no soc',
                             'HSE Fermi energy without spin-orbit coupling',
                             'eV'),
        'efermi_hse_soc': ('HSE Fermi energy - soc',
                           'HSE Fermi energy with spin-orbit coupling',
                           'eV'),
    }
    key_descriptions.update(kd)

    return kvp, key_descriptions, data


group = 'property'
resources = '24:10h'
creates = ['hse_nowfs.gpw', 'hse-restart.json']
dependencies = ['asr.structureinfo', 'asr.gs']
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart


if __name__ == '__main__':
    with cleanup('hse.gpw'):
        main.cli()
