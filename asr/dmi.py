from asr.core import command, option, DictStr, ASRResult, prepare_result
from typing import List
import numpy as np
from os import path


def findOrthoNN(kpts: List[float], pbc: List[bool], n: int = 2):
    # Warning, might not find inversion symmetric points if k-points are not symmetric
    from scipy.spatial import cKDTree

    _, indices = cKDTree(kpts).query([0, 0, 0], k=len(kpts))
    indices = indices[1:]
    N = sum(pbc)
    orthNN = [[], [], []][:N]

    for direction in np.arange(N):
        orthoDirs = [(direction + 1) % 3, (direction + 2) % 3]
        i = 0
        for j, idx in enumerate(indices):
            if np.isclose(kpts[idx][orthoDirs[0]], 0) and \
               np.isclose(kpts[idx][orthoDirs[1]], 0):
                orthNN[direction].append(kpts[idx])
                i += 1
                if i == n:
                    break

    assert 0 not in np.shape(orthNN), 'No k-points in some orthogonal direction(s)'
    assert np.shape(orthNN)[-2] == n, 'Missing k-points in orthogonal directions'
    return np.round(np.array(orthNN), 16)


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    spiraltable = table(row, 'Property', ['bandwidth', 'minimum'], key_descriptions)

    panel = {'title': 'Spin spirals',
             'columns': [[fig('spin_spiral_bs.png')], [spiraltable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['spin_spiral_bs.png']}],
             'sort': 3}
    return [panel]


@prepare_result
class PreResult(ASRResult):
    Q: np.ndarray
    key_descriptions = {"Q": "nearest neighbour orthogonal q-vectors"}
    formats = {"ase_webpanel": webpanel}


@command(module='asr.dmi',
         resources='40:1h',
         requires=['structure.json'])
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
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
        'charge': 0}) -> PreResult:
    from ase.io import read
    from ase.dft.kpoints import kpoint_convert, monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.utils.spinspiral import spinspiral
    atoms = read('structure.json')
    
    size, offsets = kpts2sizeandoffsets(atoms=atoms, **calculator['kpts'])
    kpts_kc = monkhorst_pack(size) + offsets
    kpts_kv = kpoint_convert(atoms.cell, skpts_kc=kpts_kc)
    qpts_qv = findOrthoNN(kpts_kv, atoms.pbc, n=2)
    qpts_Rqc = 2 * kpoint_convert(atoms.cell, ckpts_kv=qpts_qv)

    for i, q_qc in enumerate(qpts_Rqc):
        for j, q_c in enumerate(q_qc):
            if not path.isfile(f'gsq{j}d{i}.gpw'):
                calculator['name'] = 'gpaw'
                calculator['txt'] = f'gsq{j}d{i}.txt'
                calculator['mode']['qspiral'] = q_c
                spinspiral(calculator, write_gpw=True)
    return PreResult.fromdata(Q=qpts_Rqc)


@prepare_result
class Result(ASRResult):
    Q: np.ndarray
    dmi: np.ndarray
    key_descriptions = {"Q": "nearest neighbour orthogonal q-vectors",
                        "dmi": "Components of projected soc in orthogonal directions"}
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
    qpts_Rqc = read_json('results-asr.dmi@prepare_dmi.json')['Q']
    
    dmi_Rr = []
    for i, q_qc in enumerate(qpts_Rqc):
        en_q = []
        for j, q_c in enumerate(q_qc):
            calc = get_calculator_class(name)(f'gsq{j}d{i}.gpw')

            Ex, Ey, Ez = (soc_eigenstates(calc, projected=True,
                                          theta=theta, phi=phi).calculate_band_energy()
                          for theta, phi in [(90, 0), (90, 90), (0, 0)])
            en_q.append(np.array([Ex, Ey, Ez]))
        en_q = np.array(en_q)
        dmi_Rr.append(en_q[::2] - en_q[1::2])
    dmi_Rr = np.array(dmi_Rr)

    return Result.fromdata(Q=qpts_Rqc, dmi=dmi_Rr)


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    data = row.data.get('results-asr.spinspiral.json')
    path = data['path']
    energies = data['energies']

    energies = ((energies - energies[0]) * 1000)  # / nmagatoms
    q, x, X = path.get_linear_kpoint_axis()

    total_magmoms = data['total_magmoms']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Setup main energy plot
    ax1.plot(q, energies, c='C1', marker='.', label='Energy')
    ax1.set_ylim([np.min(energies * 1.1), np.max(energies * 1.15)])
    ax1.set_ylabel('Spin spiral energy [meV]')

    ax1.set_xlabel('q vector [Å$^{-1}$]')
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.replace('G', r"$\Gamma$") for i in X])
    for xc in x:
        if xc != min(q) and xc != max(q):
            ax1.axvline(xc, c='gray', linestyle='--')
    ax1.margins(x=0)

    # Add spin wavelength axis
    def tick_function(X):
        lmda = 2 * np.pi / X
        return [f"{z:.1f}" for z in lmda]

    # Non-cumulative length of q-vectors to find wavelength
    Q = np.linalg.norm(2 * np.pi * path.cartesian_kpts(), axis=-1)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    idx = round(len(Q) / 5)

    ax2.set_xticks(q[::idx])
    ax2.set_xticklabels(tick_function(Q[::idx]))
    ax2.set_xlabel(r"Wave length $\lambda$ [Å]")

    # Add the magnetic moment plot
    ax3 = ax1.twinx()
    mT = abs(total_magmoms[:, 0])
    # mT = np.linalg.norm(total_magmoms, axis=-1)#mT[:, 1]#
    mT2 = abs(total_magmoms[:, 1])
    mT3 = abs(total_magmoms[:, 2])
    ax3.plot(q, mT, c='r', marker='.', label='$m_x$')
    ax3.plot(q, mT2, c='g', marker='.', label='$m_y$')
    ax3.plot(q, mT3, c='b', marker='.', label='$m_z$')

    ax3.set_ylabel(r"Total norm magnetic moment ($\mu_B$)")
    mommin = np.min(mT * 0.9)
    mommax = np.max(mT * 1.15)
    ax3.set_ylim([mommin, mommax])

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # fig.suptitle('')
    plt.tight_layout()
    plt.savefig(fname)

    # energies = energies - energies[0]
    # energies = (energies)*1000
    # bs = BandStructure(path=path, energies=energies[None, :, None])
    # bs.plot(ax=plt.gca(), ls='-', marker='.', colors=['C1'],
    #         emin=np.min(energies * 1.1), emax=np.max([np.max(energies * 1.15)]),
    #         ylabel='Spin spiral energy [meV]')
    # plt.tight_layout()
    # plt.savefig(fname)


if __name__ == '__main__':
    main.cli()
