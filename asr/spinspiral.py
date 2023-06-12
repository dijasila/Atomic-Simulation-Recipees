from asr.core import command, option, argument, ASRResult, prepare_result
from asr.utils.symmetry import c2db_symmetry_eps
from typing import List, Union
import numpy as np
from os import path


def get_magnetic_moments(atoms, params, q_c, smooth):
    try:
        magmoms = params["experimental"]["magmoms"]
    except KeyError:
        if atoms.has('initial_magmoms'):
            magmomx = atoms.get_initial_magnetic_moments()
        else:
            magmomx = np.ones(len(atoms), float)
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 0] = magmomx

    if smooth:  # Smooth spiral

        from ase.dft.kpoints import kpoint_convert
            
        def rotation_matrix(axis, theta):
            """
            Return the rotation matrix associated with counterclockwise rotation
            about the given axis by theta radians.
            https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
            """
            axis = np.asarray(axis)
            axis = axis / np.sqrt(np.dot(axis, axis))
            a = np.cos(theta / 2.0)
            b, c, d = -axis * np.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                             [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                             [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        q_v = kpoint_convert(atoms.get_cell(), skpts_kc=[q_c])[0]
        pos_av = atoms.get_positions()
        theta = np.dot(pos_av, q_v)
        R = [rotation_matrix([0, 0, 1], theta[i]) for i in range(len(atoms))]
        magmoms = [R[i] @ magmoms[i] for i in range(len(atoms))]
        magmoms = np.asarray(magmoms)
    return magmoms


@command(module='asr.spinspiral',
         requires=['structure.json'])
@argument('q_c', type=List[float])
@option('--n', type=int)
@option('--params', help='Calculator parameter dictionary', type=dict)
@option('--smooth', help='Rotate initial magmoms by q dot a', type=bool)
def calculate(q_c : List[float] = [1 / 3, 1 / 3, 0], n : int = 0,
              params: dict = dict(mode={'name': 'pw', 'ecut': 400},
                                  kpts={'density': 4.0, 'gamma': True}),
              smooth: bool = True) -> ASRResult:
    """Calculate the groundstate of a given spin spiral vector q_c"""
    from ase.io import read
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    atoms = read('structure.json')
    restart = path.isfile(f'gsq{n}.gpw')

    # IF the calculation has been run (gs.txt exist) but did not converge
    # (gs.gpw not exist) THEN raise exception UNLESS it did not finish (last calc)
    # (gsn.txt exist but gsn+1.txt does not)
    # Note, UNLESS imply negation in context of if-statement (and not)
    if path.isfile(f'gsq{n}.txt') and not path.isfile(f'gsq{n}.gpw') and \
       not (path.isfile(f'gsq{n}.txt') and not path.isfile(f'gsq{n+1}.txt')):
        raise Exception("SFC finished but didn't converge")

    magmoms = get_magnetic_moments(atoms, params, q_c, smooth)

    if restart:
        params = dict(restart=f'gsq{n}.gpw')
    else:
        # Mandatory spin spiral parameters
        params["mode"]["qspiral"] = q_c
        params["xc"] = 'LDA'
        params["experimental"] = {'magmoms': magmoms, 'soc': False}
        params["symmetry"] = 'off'
        params["parallel"] = {'domain': 1, 'band': 1}
        params["txt"] = f'gsq{n}.txt'

    calc = GPAW(**params)
    # atoms.center(vacuum=4.0, axis=2)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    totmom_v, magmom_av = calc.density.estimate_magnetic_moments()
    gap, k1, k2 = bandgap(calc)

    if not restart:
        atoms.calc.write(f'gsq{n}.gpw')
    return ASRResult.fromdata(en=energy, q=q_c, ml=magmom_av, mT=totmom_v, gap=gap)


@prepare_result
class Result(ASRResult):
    path: np.ndarray
    energies: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minimum: np.ndarray
    gaps: np.ndarray
    gapmin: float
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Potential energy [eV]",
                        "local_magmoms": "List of estimated local moments [mu_B]",
                        "total_magmoms": "Estimated total moment [mu_B]",
                        "bandwidth": "Energy difference [meV]",
                        "minimum": "Q-vector at energy minimum",
                        "gaps": "List of bandgaps",
                        "gapmin": "Bandgap at minimum energy"}


@command(module='asr.spinspiral',
         requires=['structure.json'],
         returns=Result)
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--n', type=int)
@option('--params', help='Calculator parameter dictionary', type=dict)
@option('--smooth', help='Rotate initial magmoms by q dot a', type=bool)
@option('--clean_up', help='Remove gpw files after execution', type=bool)
@option('--eps', help='Bandpath symmetry threshold', type=float)
def main(q_path: Union[str, None] = None, n: int = 11,
         params: dict = dict(mode={'name': 'pw', 'ecut': 800},
                             kpts={'density': 6.0, 'gamma': True}),
         smooth: bool = True, clean_up: bool = False, eps: float = None) -> Result:
    from ase.io import read
    atoms = read('structure.json')
    cell = atoms.cell
    if eps is None:
        eps = c2db_symmetry_eps

    if q_path is None:
        # Input --q_path None
        # eps = 0.1 is current c2db threshold
        path = atoms.cell.bandpath(npoints=n, pbc=atoms.pbc, eps=eps)
        Q = np.round(path.kpts, 16)
    elif q_path == 'ibz':
        # Input: --q_path 'ibz'
        from gpaw.symmetry import atoms2symmetry
        from gpaw.kpt_descriptor import kpts2sizeandoffsets
        from ase.dft.kpoints import monkhorst_pack, kpoint_convert
        # Create (n,n,1) for 2D, (n,n,n) for 3D
        sizeInput = atoms.get_pbc() * (n - 1) + 1
        size, offset = kpts2sizeandoffsets(sizeInput, gamma=True, atoms=atoms)
        bzk_kc = monkhorst_pack(size) + offset
        symmetry = atoms2symmetry(atoms, tolerance=eps)
        ibzinfo = symmetry.reduce(bzk_kc)
        Q = ibzinfo[0]
        bz2ibz_k, ibz2bz_k = ibzinfo[4:6]
        bzk_kv = kpoint_convert(cell, skpts_kc=bzk_kc) / (2 * np.pi)
        path = [bzk_kv, bz2ibz_k, ibz2bz_k]
    elif q_path.isalpha():
        # Input: --q_path 'GKMG'
        path = atoms.cell.bandpath(q_path, npoints=n, pbc=atoms.pbc, eps=eps)
        Q = np.round(path.kpts, 16)
    else:
        # Input: --q_path 111 --n 5
        from ase.dft.kpoints import (monkhorst_pack, kpoint_convert)
        import sys
        sys.path.insert(0, "/home/niflheim/joaso/scripts/")
        from rotation import project_to_plane
        plane = [eval(q) for q in q_path]
        Q = monkhorst_pack([n, n, n])
        Q = project_to_plane(Q, plane)
        Qv = kpoint_convert(atoms.get_cell(), skpts_kc=Q) / (2 * np.pi)
        path = [Q, Qv]

    energies = []
    lmagmom_av = []
    Tmagmom_v = []
    gaps = []
    for i, q_c in enumerate(Q):
        try:
            result = calculate(q_c=q_c, n=i, params=params, smooth=smooth)
            energies.append(result['en'])
            lmagmom_av.append(result['ml'])
            Tmagmom_v.append(result['mT'])
            gaps.append(result['gap'])
        except Exception as e:
            print('Exception caught: ', e)
            energies.append(0)
            lmagmom_av.append(np.zeros((len(atoms), 3)))
            Tmagmom_v.append(np.zeros(3))
            gaps.append(0)

    energies = np.asarray(energies)
    lmagmom_av = np.asarray(lmagmom_av)
    Tmagmom_v = np.asarray(Tmagmom_v)
    gaps = np.asarray(gaps)

    bandwidth = (np.max(energies) - np.min(energies)) * 1000
    emin_idx = np.argmin(energies)
    qmin = Q[emin_idx]
    gapmin = gaps[emin_idx]
    if clean_up:
        from gpaw import mpi
        import os
        from glob import glob
        gpw_list = glob('*.gpw')
        for gpw in gpw_list:
            if int(gpw[3:-4]) != emin_idx:
                if mpi.world.rank == 0:
                    os.remove(gpw)

    return Result.fromdata(path=path, energies=energies, minimum=qmin,
                           local_magmoms=lmagmom_av, total_magmoms=Tmagmom_v,
                           bandwidth=bandwidth, gaps=gaps, gapmin=gapmin)


if __name__ == '__main__':
    main.cli()
