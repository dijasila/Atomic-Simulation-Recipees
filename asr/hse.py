"""
to do:
- better interpolation scheme?
- find reasonable default values for params
- move stuff to utils
- get evac
- create tests
- move relevant functions to hseinterpolate? or merge into one single recipe?
- get spin_axis, get_spin_direction, eigenvalues from asr.utils.gpw2eigs
"""
import json
from pathlib import Path
from asr.core import command, option, read_json
from asr.utils.gpw2eigs import eigenvalues, get_spin_direction, fermi_level

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

@command(module='asr.hse',
         dependencies = ['asr.structureinfo', 'asr.gs'],
         creates=['hse_nowfs.gpw', 'hse-restart.json'],
         #tests=...,
         requires=['gs.gpw'],
         resources='24:10h',
         restart=2)
@option('--kptdensity', help='K-point density')
@option('--emptybands', help='number of empty bands to include')
def calculate(kptdensity=12, emptybands=20):
    """Calculate HSE band structure"""
    with cleanup('hse.gpw'):
        results = {}
        results['hse_eigenvalues'] = hse(kptdensity=kptdensity, emptybands=emptybands)
        mpi.world.barrier()
        results['hse_eigenvalues_soc'] = hse_spinorbit(results['hse_eigenvalues'])
        return results

@command(module='asr.hse',
         dependencies = ['asr.hse@calculate', 'asr.bandstructure'],
         #tests=...,
         requires=['hse_nowfs.gpw', 'results-asr.bandstructure.json', 'bs.gpw'],
         resources='8:10m',
         restart=2)
@option('--kptpath', type=str)
@option('--npoints')
def main(kptpath=None, npoints=400):
    """Interpolate HSE band structure along a given path"""
    results = MP_interpolate(kptpath, npoints)
    return results


def hse(kptdensity, emptybands):

    convbands = int(emptybands / 2)
    if not os.path.isfile('hse.gpw'):
        calc = GPAW('gs.gpw', txt=None)
        atoms = calc.get_atoms()
        pbc = atoms.pbc.tolist()
        ND = np.sum(pbc)
        if ND == 3 or ND == 1:
            kpts = {'density': kptdensity, 'gamma': True, 'even': False} # do not round up to nearest even number!
        elif ND == 2:

            # XXX move to utils? [also in asr.polarizability]
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

def MP_interpolate(kptpath, npoints=400, show=False):
    """
    Calculates band stucture along the same band path used for PBE.
    Band structure is obtained by using 'monkhorst_pack_interpolate' to get the HSE correction
    """
    # read PBE (without SOC)
    results_bandstructure = read_json('results-asr.bandstructure.json')  
    path = results_bandstructure['bs_nosoc']['path']
    e_pbe_skn = results_bandstructure['bs_nosoc']['energies']

    # get the HSE correction to PBE eigenvalues (delta_skn)
    calc = GPAW('hse_nowfs.gpw', txt=None)
    atoms = calc.atoms
    results_hse = read_json('results-asr.hse@calculate.json')
    data = results_hse['hse_eigenvalues']
    nbands = results_hse['hse_eigenvalues']['e_hse_skn'].shape[2]
    delta_skn = data['vxc_hse_skn']-data['vxc_pbe_skn']
    delta_skn.sort(axis=2)

    # interpolate delta_skn to kpts along the band path
    size, offset = get_monkhorst_pack_size_and_offset(calc.get_bz_k_points())
    bz2ibz = calc.get_bz_to_ibz_map()
    icell = atoms.get_reciprocal_cell()
    eps = monkhorst_pack_interpolate(path.kpts, delta_skn.transpose(1, 0, 2),
                                     icell, bz2ibz, size, offset)
    delta_interp_skn = eps.transpose(1, 0, 2)
    e_hse_skn = e_pbe_skn[:,:,:nbands] + delta_interp_skn
    dct = dict(e_hse_skn=e_hse_skn, path=path)

    # add SOC from bs.gpw
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    if mpi.world.rank in ranks:
        calc = GPAW('bs.gpw', communicator=comm, txt=None)
        theta, phi = get_spin_direction()
        e_hse_mk, s_hse_mk = get_soc_eigs(calc, gw_kn=e_hse_skn, return_spin=True,
                                          bands=np.arange(e_hse_skn.shape[2]),
                                          theta=theta, phi=phi)
        dct.update(e_hse_mk=e_hse_mk, s_hse_mk=s_hse_mk)

    results = {}
    results['hse_bandstructure'] = dct

    return results

# XXX move to utils? [also in asr.bandstructure] -> in asr.utils.gpw2eigs (?)
def spin_axis(fname='anisotropy_xy.npz') -> int:
    import numpy as np
    theta, phi = get_spin_direction(fname=fname)
    if theta == 0:
        return 2
    elif np.allclose(phi, np.pi / 2):
        return 1
    else:
        return 0

# XXX move to utils?
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

    if not os.path.isfile('results-asr.hse@calculate.json'):
        return kvp, key_descriptions, data

    results_hse = read_json('results-asr.hse@calculate.json')
    
    eps_skn = results_hse['hse_eigenvalues']['e_hse_skn']
    calc = GPAW('hse_nowfs.gpw', txt=None)
    ibzkpts = calc.get_ibz_k_points()

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


if __name__ == '__main__':
    main.cli()
