from __future__ import annotations
from asr.core import command, option, ASRResult, prepare_result, DictStr
from asr.utils.spinspiral import extract_magmoms, rotate_magmoms, \
    get_noncollinear_magmoms, get_spiral_bandpath
from asr.collect_spiral import SpinSpiralCalculation
from ase.io import read
from ase.parallel import world
from typing import Union
import numpy as np
from os import path


def spinspiral(calculator: dict = {
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'experimental': {'soc': False},
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'density': 12.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gsq.txt',
        'charge': 0},
        atoms=None, rotation_model='q.a',
        write_gpw: bool = True) -> dict:
    """Calculate the groundstate of a given spin spiral vector q_c"""
    from gpaw import GPAW
    from ase.dft.bandgap import bandgap

    if atoms is None:
        atoms = read('structure.json')

    try:
        gpwfile = calculator['txt'].replace('.txt', '.gpw')
        restart = path.isfile(gpwfile)
    except KeyError:
        gpwfile = 'gsq.gpw'
        restart = False

    q_c = calculator['mode']['qspiral']  # spiral vector must be provided
    magmoms = extract_magmoms(atoms, calculator)
    if rotation_model is not None:
        magmoms = rotate_magmoms(atoms, magmoms, q_c, model=rotation_model)

    # Mandatory spin spiral parameters
    calculator["xc"] = 'LDA'
    calculator["experimental"] = {'magmoms': magmoms, 'soc': False}
    calculator["symmetry"] = 'off'
    calculator["parallel"] = {'domain': 1, 'band': 1}

    if restart:
        calc = GPAW(gpwfile)
    else:
        calc = GPAW(**calculator)

    # atoms.center(vacuum=6.0, axis=2)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    totmom_v, magmom_av = calc.density.estimate_magnetic_moments()
    gap, _, _ = bandgap(calc)

    if write_gpw and not restart:
        atoms.calc.write(gpwfile)
    return {'energy': energy, 'totmom_v': totmom_v,
            'magmom_av': magmom_av, 'gap': gap}


@prepare_result
class Result(ASRResult):
    path: np.ndarray
    energies: np.ndarray
    key_descriptions = {"path": "List of Spin spiral vectors",
                        "energies": "Potential energy [eV]"}


@command(module='asr.spinspiral',
         requires=['structure.json'],
         returns=Result)
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--qpts', help='Number of sample points', type=int)
@option('--qdens', help='Density of sample points', type=float)
@option('--rotation_model', help='Rotate initial magmoms by given model', type=str)
@option('--clean_up', help='Remove gpw files after execution', type=bool)
@option('--eps', help='Bandpath symmetry threshold', type=float)
def main(calculator: dict = dict(mode={'name': 'pw', 'ecut': 800},
                                 kpts={'density': 6.0, 'gamma': True}),
         q_path: Union[str, None] = None,
         qpts: int = None, qdens: float = None,
         rotation_model: Union[str, None] = 'q.a', clean_up: bool = False,
         eps: float = None) -> Result:

    atoms = read('structure.json')

    q_path = get_spiral_bandpath(atoms=atoms, qdens=qdens, qpts=qpts,
                                 q_path=q_path, eps=eps)

    try:
        magmoms_b = [calculator["experimental"]["magmoms"]]
    except KeyError:
        magmoms_b = get_noncollinear_magmoms(atoms)

    energies = []
    for bidx, magmoms in enumerate(magmoms_b):
        for qidx, qn_c in enumerate(q_path.kpts):
            calculator['mode']['qspiral'] = qn_c
            calculator['txt'] = f'gsq{qidx}b{bidx}.txt'
            calculator['experimental']['magmoms'] = magmoms

            if cannot_converge(qidx, bidx):
                continue

            try:
                result = spinspiral(calculator=calculator, atoms=atoms,
                                    rotation_model=rotation_model)
                energies.append(result['energy'])
                sscalc = SpinSpiralCalculation([bidx, qidx],
                                               result['energy'],
                                               result['totmom_v'].tolist(),
                                               result['magmom_av'].tolist(),
                                               result['gap'])
            except Exception as e:
                print(f'Caught exception {e}')
                energies.append(0.0)
                sscalc = SpinSpiralCalculation([bidx, qidx], 0.0, [0, 0, 0],
                                               [[0, 0, 0]] * len(atoms), 0)
            if world.rank == 0:
                sscalc.save(f'datq{qidx}b{bidx}.json')

    energies = np.asarray(energies)
    emin_idx = np.argmin(energies)
    if clean_up:
        import os
        from glob import glob
        gpw_list = glob('*.gpw')
        for gpw in gpw_list:
            if int(gpw[3:-6]) != emin_idx:
                if world.rank == 0:
                    os.remove(gpw)

    return Result.fromdata(path=q_path, energies=energies)


def cannot_converge(qidx, bidx):
    # IF the calculation has been run (gs.txt exist) but did not converge
    # (gs.gpw not exist) THEN raise exception UNLESS it did not finish (last calc)
    # (gsn.txt exist but gsn+1.txt does not)
    # Note, UNLESS imply negation in context of if-statement (and not)
    if path.isfile(f'gsq{qidx}b{bidx}.txt') and \
        not path.isfile(f'gsq{qidx}b{bidx}.gpw') and \
        not (path.isfile(f'gsq{qidx}b{bidx}.txt') and not
             path.isfile(f'gsq{qidx+1}b{bidx}.txt')):
        return True  # SFC finished but didn't converge
    return False
