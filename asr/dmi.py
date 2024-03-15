from asr.core import command, option, DictStr, ASRResult, prepare_result
from typing import List
import numpy as np
from os import path


def find_ortho_nn(kpts: List[float],
                  pbc: List[bool],
                  npoints=[2, 2, 2],
                  eps: float = 0):
    '''
    Given a list of kpoints, we find the points along vectors [1,0,0], [0,1,0], [0,0,1]
    and search through them ordered on the distance to the origin. Vectors along the
    postive axis will appear first.
    '''

    if isinstance(npoints, int):
        npoints = [npoints] * 3

    # Warning, might not find inversion symmetric points if k-points are not symmetric
    from scipy.spatial import cKDTree

    # Calculate distance-ordered indices from the (eps postive) origin
    _, indices = cKDTree(kpts).query([eps, eps, eps], k=len(kpts))
    indices = indices[1:]
    periodic_directions = np.where(pbc)[0]

    orthNN = []
    for direction in periodic_directions:
        xNN = find_neighbours_in_line(kpts, direction, indices, npoints)

        if len(xNN) > 0:
            xNN = np.round(np.array(xNN), 16)
            orthNN.append(xNN)

    shape = [np.shape(orthNN[j]) for j in range(len(orthNN))]
    print('shape', shape)
    assert (0,) not in shape, \
        f'No k-points in some periodic direction(s), out.shape = {shape}'
    assert shape != [], 'No k-points were found'
    return orthNN


def kgrid_to_qgrid(k_qc):
    # Choose q=2k and cutoff points at BZ edge
    q_qc = 2 * k_qc
    q_qc = q_qc[np.linalg.norm(q_qc, axis=-1) <= 0.5]
    even_q = int(np.floor(len(q_qc) / 2) * 2)
    q_qc = q_qc[:even_q]
    return q_qc


def find_neighbours_in_line(kpts, direction, indices, npoints):
    orthNN = []
    orthoDirs = [(direction + 1) % 3, (direction + 2) % 3]
    i = 0
    for j, idx in enumerate(indices):
        # Check if point lies on a line x, y or z
        if np.isclose(kpts[idx][orthoDirs[0]], 0) and \
           np.isclose(kpts[idx][orthoDirs[1]], 0):
            if i >= npoints[direction]:
                return orthNN
            orthNN.append(kpts[idx])
            i += 1
    return orthNN


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
@option('-n', help='Number of points along orthogonal directions', type=int)
def prepare_dmi(calculator: dict = {
        'mode': {'name': 'pw', 'ecut': 800, 'qspiral': [0, 0, 0]},
        'xc': 'LDA',
        'soc': False,
        'symmetry': 'off',
        'parallel': {'domain': 1, 'band': 1},
        'kpts': {'size': (32, 32, 1), 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'txt': 'gsq.txt',
        'charge': 0}, n: int = 2) -> ASRResult:
    from ase.io import read
    from ase.dft.kpoints import monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.spinspiral import spinspiral
    from gpaw.new.ase_interface import GPAW
    atoms = read('structure.json')

    size, offsets = kpts2sizeandoffsets(atoms=atoms, **calculator['kpts'])
    kpts_kc = monkhorst_pack(size) + offsets
    kpts_nqc = find_ortho_nn(kpts_kc, atoms.pbc, npoints=n)
    en_nq = []
    q_nqc = []
    for i, k_qc in enumerate(kpts_nqc):
        en_q = []

        q_qc = kgrid_to_qgrid(k_qc)
        for j, q_c in enumerate(q_qc):
            if not path.isfile(f'gsq{j}d{i}.gpw'):
                calculator['txt'] = f'gsq{j}d{i}.txt'
                calculator['mode']['qspiral'] = q_c
                result = spinspiral(calculator)
                en_q.append(result['energy'])
            else:
                result = GPAW(f'gsq{j}d{i}.gpw')
                en_q.append(result['energy'])
        en_nq.append(en_q)
        q_nqc.append(q_qc)
    return PreResult.fromdata(qpts_nqc=q_nqc, en_nq=en_nq)


def webpanel(result, row, key_descriptions):
    D_nqv = result.get('D_nqv')
    D_nv = []
    for D_qv in D_nqv:
        D_nv.append(D_qv[0])
    D_nv = np.round(np.array(D_nv), 2) + 0.

    # Estimate error
    en_nq = result.get('en_nq')
    abs_error_threshold = 1.1e-7
    rel_error_threshold = 1.1e-8

    error_marker = []
    for n in range(len(en_nq)):
        abs_error = en_nq[n][0] - en_nq[n][1]
        try:
            rel_error = abs_error / en_nq[n][0]
        except ZeroDivisionError:
            rel_error = 0

        if abs(abs_error) > abs_error_threshold or \
           abs(rel_error) > rel_error_threshold:
            error_marker.append('*')
        else:
            error_marker.append('')

    formatter = {'float': lambda x: f'{x:.2f}'}
    rows = [['D(q<sub>' + ['a1', 'a2', 'a3'][n] + '</sub>) (meV / Å<sup>-1</sup>)',
             f"{np.array2string(D_nv[n], formatter=formatter)}{error_marker[n]}"]
            for n in range(len(D_nv))]

    dmi_table = {'type': 'table',
                 'header': ['Property', 'Value'],
                 'rows': rows}

    panel = {'title': 'Spin spirals',
             'columns': [[], [dmi_table]],
             'sort': 2}
    return [panel]


@prepare_result
class Result(ASRResult):
    en_nq: List
    en_soc_nq: List
    qpts_nqc: List
    qpts_soc_nqc: List
    D_nqv: List[np.ndarray]
    DMI: float
    key_descriptions = {"en_nq": "Energy of spin spiral groundstates",
                        "en_soc_nq": "Antisymmetric Dzyaloshinskii moriya energy",
                        "qpts_nqc": "q-vectors of spin spiral groundstates",
                        "qpts_soc_nqc": "Sign corrected q-vectors of DM",
                        "D_nqv": "Components of effective DM vector [meV Å]",
                        "DMI": 'Dzyaloshinskii Moriya interaction [meV Å]'}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.dmi',
         dependencies=['asr.dmi@prepare_dmi'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    from gpaw.new.ase_interface import GPAW
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.occupations import create_occ_calc
    from ase.dft.kpoints import kpoint_convert
    from ase.io import read
    from asr.core import read_json
    atoms = read('structure.json')

    qpts_nqc = read_json('results-asr.dmi@prepare_dmi.json')['qpts_nqc']
    en_nq = read_json('results-asr.dmi@prepare_dmi.json')['en_nq']
    width = 0.001
    occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': width})

    E_nq = []
    q_nqc = []
    D_nqv = []
    for n, q_qc in enumerate(qpts_nqc):
        Esoc_q = []
        for j, q_c in enumerate(q_qc):
            calc = f'gsq{j}d{n}.gpw'
            calc = GPAW(calc)
            Ex, Ey, Ez = (soc_eigenstates(calc, projected=True, occcalc=occcalc,
                                          theta=th, phi=phi).calculate_band_energy()
                          for th, phi in [(90, 0), (90, 90), (0, 0)])
            # This is required, otherwise result is very noisy
            E0x, E0y, E0z = (soc_eigenstates(calc, projected=True,
                                             occcalc=occcalc, scale=0,
                                             theta=th, phi=phi).calculate_band_energy()
                             for th, phi in [(90, 0), (90, 90), (0, 0)])
            Esoc_q.append(np.array([Ex - E0x, Ey - E0y, Ez - E0z]))
        Esoc_q = np.array(Esoc_q)
        q_qc = (q_qc[::2] - q_qc[1::2]) / 2
        E_q = (Esoc_q[::2] - Esoc_q[1::2]) / 2

        # Sign correction
        sign_correction_q = np.sign(np.sum(q_qc, axis=-1))
        q_qc = (q_qc.T * sign_correction_q).T
        E_q = (E_q.T * sign_correction_q).T

        dq_qv = kpoint_convert(cell_cv=atoms.cell, skpts_kc=q_qc)
        dq_q = np.linalg.norm(dq_qv, axis=-1)
        D_qv = np.divide(-2 * 1000 * E_q.T, dq_q).T

        # Append to data
        E_nq.append(E_q)
        q_nqc.append(q_qc)
        D_nqv.append(D_qv)

    # Sortable key-value pair
    DMI = np.round(np.max(np.linalg.norm(D_nqv[0][0], axis=-1)), 2)
    return Result.fromdata(qpts_nqc=qpts_nqc, en_nq=en_nq,
                           qpts_soc_nqc=q_nqc, en_soc_nq=E_nq,
                           D_nqv=D_nqv, DMI=DMI)


if __name__ == '__main__':
    main.cli()
