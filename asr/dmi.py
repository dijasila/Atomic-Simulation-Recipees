from asr.core import command, option, DictStr, ASRResult, prepare_result
from typing import List
import numpy as np
from os import path


def findOrthoNN(kpts: List[float], pbc: List[bool], n: int = 2, eps: float = 0):
    '''
    Given a list of kpoints, we find the points along vectors [1,0,0], [0,1,0], [0,0,1]
    and search through them ordered on the distance to the origin. Vectors along the
    postive axis will appear first.
    '''
    # Warning, might not find inversion symmetric points if k-points are not symmetric
    from scipy.spatial import cKDTree

    # Calculate distance-ordered indices from the (eps postive) origin
    _, indices = cKDTree(kpts).query([eps, eps, eps], k=len(kpts))
    indices = indices[1:]

    N = sum(pbc)
    orthNN = [[], [], []][:N]
    for direction in np.arange(N):
        orthoDirs = [(direction + 1) % 3, (direction + 2) % 3]
        i = 0
        for j, idx in enumerate(indices):
            # Check if point lies on a line x, y or z
            if np.isclose(kpts[idx][orthoDirs[0]], 0) and \
               np.isclose(kpts[idx][orthoDirs[1]], 0):
                orthNN[direction].append(kpts[idx])
                i += 1
                if i == n:
                    break

    shape = [np.shape(orthNN[i]) for i in range(N)]
    assert (0,) not in shape, \
        f'No k-points in some periodic direction(s), out.shape = {shape}'
    assert all([(n, 3) == np.shape(orthNN[i]) for i in range(N)]), \
        f'Missing k-points in some periodic direction(s), out.shape = {shape}'
    # This test is incompatible with len(pbc) = 2, while it works fine otherwise
    # assert not all([all(np.dot(orthNN[i], pbc) == 0) for i in range(N)]), \
    #     f'The k-points found are in a non-periodic direction'
    return np.round(np.array(orthNN), 16)


@prepare_result
class PreResult(ASRResult):
    qpts_nqc: np.ndarray
    en_nq: np.ndarray
    key_descriptions = {"qpts_nqc": "nearest neighbour orthogonal q-vectors",
                        "en_nq": "energy of respective spin spiral groundstates"}


@command(module='asr.dmi',
         resources='40:1h',
         requires=['structure.json'])
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('-n', help='Number of points along ortho directions', type=DictStr())
def prepare_dmi(calculator: dict = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'experimental': {'soc': False},
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'density': 22.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '200%',
        'txt': 'gsq.txt',
        'charge': 0}, n: int = 2) -> PreResult:
    from ase.io import read
    from ase.dft.kpoints import monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.utils.spinspiral import spinspiral
    from gpaw import GPAW
    atoms = read('structure.json')

    size, offsets = kpts2sizeandoffsets(atoms=atoms, **calculator['kpts'])
    kpts_kc = monkhorst_pack(size) + offsets
    qpts_nqc = 2 * findOrthoNN(kpts_kc, atoms.pbc, n=n)
    en_nq = []
    for i, q_qc in enumerate(qpts_nqc):
        en_q = []
        for j, q_c in enumerate(q_qc):
            if not path.isfile(f'gsq{j}d{i}.gpw'):
                calculator['name'] = 'gpaw'
                calculator['txt'] = f'gsq{j}d{i}.txt'
                calculator['mode']['qspiral'] = q_c
                calc = spinspiral(calculator, write_gpw=True, return_calc=True)
                en_q.append(calc.get_potential_energy())
            else:
                calc = GPAW(f'gsq{j}d{i}.gpw')
                en_q.append(calc.get_potential_energy())
        en_nq.append(en_q)
    en_nq = np.array(en_nq)
    return PreResult.fromdata(qpts_nqc=qpts_nqc, en_nq=en_nq)


def webpanel(result, row, key_descriptions):
    from ase.dft.kpoints import kpoint_convert
    qpts_nqc = result.get('qpts_nqc')
    qpts_nqv = kpoint_convert(cell_cv=row.cell, skpts_kc=qpts_nqc)
    dq = np.linalg.norm(qpts_nqv, axis=-1)
    dE = 1000 * result.get('dmi_nq')
    D_v = np.round(np.divide(dE[:, 0].T, dq[:, 0]).T, 2) + 0.

    # Estimate error
    en_nq = result.get('en_nq')    
    energy_error = (en_nq[:, 0] - en_nq[:, 1])
    print(abs(energy_error), abs(energy_error / en_nq[0]))
    abs_error_threshold = 1.1e-7
    rel_error_threshold = 1.1e-8
    error_marker = ['*' if abs(energy_error[i]) > abs_error_threshold
                    or abs(energy_error[i] / en_nq[0][i]) > rel_error_threshold
                    else '' for i in range(len(dq))]
    
    rows = [['D(q<sub>'+'xyz'[i]+'</sub>) (meV / Å<sup>-1</sup>)',
             f"{np.array2string(D_v[i], formatter={'float': lambda x: f'{x:.2f}'})}{error_marker[i]}"]
            for i in range(len(dq))]

    dmi_table = {'type': 'table',
                 'header': ['Property', 'Value'],
                 'rows': rows}

    panel = {'title': 'Spin spirals',
             'columns': [[], [dmi_table]],
             'sort': 2}
    return [panel]


@prepare_result
class Result(ASRResult):
    qpts_nqc: np.ndarray
    en_nq: np.ndarray
    dmi_nq: np.ndarray
    key_descriptions = {"qpts_nqc": "nearest neighbour orthogonal q-vectors",
                        "en_nq": "energy of respective spin spiral groundstates",
                        "dmi_nq": "Components of projected soc in orthogonal q-vectors"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.dmi',
         dependencies=['asr.dmi@prepare_dmi'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from ase.calculators.calculator import get_calculator_class
    from asr.core import read_json
    name = 'gpaw'
    qpts_nqc = read_json('results-asr.dmi@prepare_dmi.json')['qpts_nqc']
    en_nq = read_json('results-asr.dmi@prepare_dmi.json')['en_nq']

    width = 0.001
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    dmi_nq = []
    for i, q_qc in enumerate(qpts_nqc):
        en_q = []
        for j, q_c in enumerate(q_qc):
            calc = get_calculator_class(name)(f'gsq{j}d{i}.gpw')

            Ex, Ey, Ez = (soc_eigenstates(calc, projected=True, occcalc=occcalc,
                                          theta=th, phi=phi).calculate_band_energy()
                          for th, phi in [(90, 0), (90, 90), (0, 0)])
            # This is required, otherwise result is very noisy
            E0x, E0y, E0z = (soc_eigenstates(calc, projected=True,
                                             occcalc=occcalc, scale=0,
                                             theta=th, phi=phi).calculate_band_energy()
                             for th, phi in [(90, 0), (90, 90), (0, 0)])
            en_q.append(np.array([Ex - E0x, Ey - E0y, Ez - E0z]))
        en_q = np.array(en_q)
        dmi_nq.append(en_q[::2] - en_q[1::2])
    dmi_nq = np.array(dmi_nq)

    return Result.fromdata(qpts_nqc=qpts_nqc, en_nq=en_nq, dmi_nq=dmi_nq)


if __name__ == '__main__':
    main.cli()
