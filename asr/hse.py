"""
HSE band structure

Creates: hse.gpw, hse_nowfs.gpw, hse_eigenvalues.npz, hse_eigenvalues_soc.npz, hse_bandstructure.npz, hse_bandstructure3.npz
"""

"""
to do:

- restyle according to asr/asr/utils/something.py [template recipe]
- eliminate calls to C2DB
- dependencies? asr.gs, asr.anisotropy?
- should hse.gpw be removed afterwards?
  [currently both hse.gpw and hse_nowfs.gpw are kept, which is probably redundant]
- interpolation: here or in separate script (as C2DB)?
- functions _interpolate and interpolate_bandstructure are NOT used!!
- instead, interpolate_bandstructure is imported from c2db.bsinterpol as ip_bs
- substitute .npz with .json?
- set diskspace, restart
- from c2db.utils import eigenvalues -> eigenvaules function in asr.bandstructure?
- for interpolation:
  UserWarning: Please do not use (kpts, x, X) = bandpath(...). 
  Use path = bandpath(...) and then use the methods of the path object (see the BandPath class)
  otherwise you get ValueError: `x` must be strictly increasing sequence.
"""
import json
from pathlib import Path
from asr.utils import command, option

import time

from gpaw import GPAW
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw.spinorbit import get_spinorbit_eigenvalues as get_soc_eigs
import numpy as np
from ase.parallel import paropen
import os
import gpaw.mpi as mpi
from ase.dft.kpoints import (get_monkhorst_pack_size_and_offset,
                             monkhorst_pack_interpolate,
                             bandpath, parse_path_string)
#from c2db.utils import (eigenvalues, get_special_2d_path, get_spin_direction,
#                        spin_axis)
from ase.io import read
from ase.dft.kpoints import get_cellinfo
#from c2db.bsinterpol import interpolate_bandstructure as ip_bs
from contextlib import contextmanager


def main():
    hse()
    mpi.world.barrier()
    hse_spinorbit()
    mpi.world.barrier()
    # Move these to separate step as in c2db?
    #bs_interpolate()
    #mpi.world.barrier()


# move to utils? [also in asr.polarizability]
# def get_kpts_size(atoms, density):
#     """trying to get a reasonable monkhorst size which hits high
#     symmetry points
#     """
#     from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
#     size, offset = k2so(atoms=atoms, density=density)
#     size[2] = 1
#     for i in range(2):
#         if size[i] % 6 != 0:
#             size[i] = 6 * (size[i] // 6 + 1)
#     kpts = {'size': size, 'gamma': True}
#     return kpts


def hse(kptdensity=12, emptybands=20):
    if os.path.isfile('hse_eigenvalues.npz'):
        return

    convbands = int(emptybands / 2)

    if not os.path.isfile('hse.gpw'):
        calc = GPAW('gs.gpw', txt=None)
        atoms = calc.get_atoms()
        pbc = atoms.pbc.tolist()
        ND = np.sum(pbc)
        if ND == 3:
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

    if mpi.world.rank == 0:
        dct = dict(vxc_hse_skn=vxc_hse_skn,
                   e_pbe_skn=e_pbe_skn,
                   vxc_pbe_skn=vxc_pbe_skn,
                   e_hse_skn=e_hse_skn)
        with open('hse_eigenvalues.npz', 'wb') as f:
            np.savez(f, **dct)


def hse_spinorbit():
    if not os.path.isfile('hse_eigenvalues.npz'):
        return
    if not os.path.isfile('hse_nowfs.gpw'):
        return

    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('hse_nowfs.gpw', communicator=comm, txt=None)
        with open('hse_eigenvalues.npz', 'rb') as fd:
            dct = dict(np.load(fd))

        e_skn = dct.get('e_hse_skn')
        dct = {}
        theta, phi = get_spin_direction()
        e_mk, s_kvm = get_soc_eigs(calc, gw_kn=e_skn, return_spin=True,
                                   bands=np.arange(e_skn.shape[2]),
                                   theta=theta, phi=phi)
        dct['e_hse_mk'] = e_mk
        dct['s_hse_mk'] = s_kvm[:, spin_axis(), :].transpose()
        with open('hse_eigenvalues_soc.npz', 'wb') as fd:
            np.savez(fd, **dct)


def bs_interpolate(npoints=400, show=False):
    """inpolate the eigenvalues on a monkhorst pack grid to
    a path in the bz between high-symmetry points.
    The soc is added before interpolation in constrast to
    interpolate_bandstructure where the soc is added after interpolation.
    """
    calc = GPAW('hse_nowfs.gpw', txt=None)
    atoms = calc.atoms
    with paropen('hse_eigenvalues.npz', 'rb') as f:
        data = np.load(f)
        e_skn = data['e_hse_skn']
        e_skn.sort(axis=2)
    try:
        with paropen('hse_eigenvalues_soc.npz', 'rb') as f:
            data = np.load(f)
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
    path = atoms.cell.bandpath(npoints=npoints) # is it the correct usage
    str_path = path.labelseq # this returns a sequence of labels (eg 'GMKG' for 2D hexagonal lattice)
    icell = atoms.get_reciprocal_cell()
    eps = monkhorst_pack_interpolate(path.kpts, e_skn.transpose(1, 0, 2),
                                     icell, bz2ibz, size, offset) # monkhorst_pack_interpolate wants a (npoints, 3) array
    eps_skn = eps.transpose(1, 0, 2)
    dct = dict(eps_skn=eps_skn, path=path.kpts, kptpath=str_path)
    if e_mk is not None:
        eps_soc = monkhorst_pack_interpolate(path.kpts, e_mk.transpose(1, 0),
                                             icell, bz2ibz, size, offset)
        s_soc = monkhorst_pack_interpolate(path.kpts, s_mk.transpose(1, 0),
                                           icell, bz2ibz, size, offset)
        e_mk = eps_soc.transpose(1, 0)
        s_mk = s_soc.transpose(1, 0)
        dct.update(e_mk=e_mk, s_mk=s_mk)
    if mpi.world.rank in [0]:
        with paropen('hse_bandstructure.npz', 'wb') as f:
            np.savez(f, **dct)
    # use spline
    # get edge points
    cell = read('gs.gpw').cell
    special_points = path.special_points # a dictionary of special points
    kpoints = []
    for k in parse_path_string(kptpath)[0]:
        kpoints.append(special_points[k])
    kpoints = np.array(kpoints) # an array with special points coordinates

    # third time is a charm
    eps_skn = np.load('hse_eigenvalues.npz')['e_hse_skn']
    kpts, e_skn, _, _ = interpolate_bandstructure(calc, e_skn=e_skn, npoints=npoints)
    dct = dict(eps_skn=e_skn, path=kpts)

    eps_smk = np.load('hse_eigenvalues_soc.npz')['e_hse_mk']
    eps_smk = eps_smk[np.newaxis]
    kpts, e_skn, xr, yr_skn = interpolate_bandstructure(calc, e_skn=eps_smk.transpose(0, 2, 1),
                                    npoints=npoints)
    dct.update(e_mk=e_skn[0].transpose(), path=kpts, xreal=xr,
               epsreal_skn=yr_skn)
    if mpi.world.rank in [0]:
        with open('hse_bandstructure3.npz', 'wb') as fd:
            np.savez(fd, **dct)

# remove these functions
# def _interpolate(calc, kpts_kc, e_skn=None):
#     """
#     Parameters:
#         calc: Calculator
#             GPAW calcualtor
#         kpts_kc: (nk, 3)-shape array
#             kpoints to interpolate onto
#         e_skn: (ns, nk, nb)-shape array
#             array values on the kpoint grid in calc
#     Returns:
#         eps_skn: (ns ,nk, nb)-shape array
#             the array values in e_skn interpolated onto kpts_kc
#     """
#     if e_skn is None:
#         e_skn = eigenvalues(calc)
#     atoms = calc.get_atoms()
#     icell = atoms.get_reciprocal_cell()
#     bz2ibz = calc.get_bz_to_ibz_map()
#     size, offset = get_monkhorst_pack_size_and_offset(calc.get_bz_k_points())
#     eps = monkhorst_pack_interpolate(kpts_kc, e_skn.transpose(1, 0, 2),
#                                      icell, bz2ibz, size, offset)
#     return eps.transpose(1, 0, 2)

# remove this function? 
# def interpolate_bandstructure():
#     """Interpolate eigenvalues onto the kpts in bs.gpw (bandstruct calc) using the
#        eigenvalues in hse_eigenvalues.npz (which comes from hse@hse_nowfs.gpw).
#        Spin orbit coupling is calculated using the interpolated eigenvalues and
#        the projetor overlaps in bs.gpw.
#     """
#     ranks = [0]
#     comm = mpi.world.new_communicator(ranks)
#     if mpi.world.rank in ranks:
#         calc1 = GPAW('bs.gpw', txt=None, communicator=comm)
#         calc2 = GPAW('hse_nowfs.gpw', txt=None, communicator=comm)
#         e_hse_skn = np.load('hse_eigenvalues.npz')['e_hse_skn']
#         e_hse_skn.sort(axis=2)  # this will prob give problems with s_mk
#         path = calc1.get_ibz_k_points()
#         eps_hse_skn = _interpolate(calc=calc2,
#                                    kpts_kc=path,
#                                    e_skn=e_hse_skn)
#         eps_skn = eps_hse_skn
#         theta, phi = get_spin_direction()
#         e_mk, s_kvm = get_soc_eigs(calc1, gw_kn=eps_skn, return_spin=True,
#                                    bands=range(eps_skn.shape[2]),
#                                    theta=theta, phi=phi)
#         s_mk = s_kvm.transpose(1, 2, 0)[spin_axis()]
#         dct = dict(e_mk=e_mk, s_mk=s_mk, eps_skn=eps_hse_skn, path=path)
#         with open('hse_bandstructure.npz', 'wb') as fd:
#             np.savez(fd, **dct)

# move to utils?
def interpolate_bandstructure(calc, e_skn=None, npoints=400):
    """simple wrapper for interpolate_bandlines2
    Returns:
        out: kpts, e_skn, xreal, epsreal_skn
    """
    #path = get_special_2d_path(cell=calc.atoms.cell) # no need for that! you can get from calc.atoms.cell.bandpath(npoints=npoints)
    r = interpolate_bandlines2(calc=calc, e_skn=e_skn, npoints=npoints)
    return r['kpts'], r['e_skn'], r['xreal'], r['epsreal_skn']

# move to utils?
def interpolate_bandlines2(calc, e_skn=None, npoints=400):
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
    if e_skn is None:
        e_skn = eigenvalues(calc)
    kpts = calc.get_bz_k_points()
    bz2ibz = calc.get_bz_to_ibz_map()
    cell = calc.atoms.cell
    indices, x = segment_indices_and_x(cell=cell, path_str=path, kpts=kpts)
    # kpoints and positions to interpolate onto
    kpts2, x2, X2 = bandpath(cell=cell, path=path, npoints=npoints) ### NO! USE BandPath class instead!!
    # patu = cell.bandpath(npoints=npoints) # use this
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

if __name__ == '__main__':
    with cleanup('hse.gpw'):
        main()






########################################################

#def bs_hse(row, path):
#    from asr.gw import bs_xc
#    bs_xc(row, path, xc='hse', label='HSE')


# def webpanel(row, key_descriptions):
#     from asr.utils.custom import fig, table
#     hse = table('Property',
#                 ['work_function_hse', 'dos_hse', 'gap_hse', 'dir_gap_hse',
#                  'vbm_hse', 'cbm_hse'],
#                  key_descriptions=key_descriptions_noxc)

#     panel = ('Electronic band structure (HSE)',
#              [[fig('hse-bs.png')],
#               [hse]])

#     return panel

group = 'property'
resources = '24:10h'
creates = ['hse_nowfs.gpw',
           'hse_eigenvalues.npz',
           'hse_eigenvalues_soc.npz',
           'hse_bandstructure.npz',
           'hse_bandstructure3.npz',
           'hse-restart.json']
dependencies = ['asr.structureinfo', 'asr.gs']
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart
