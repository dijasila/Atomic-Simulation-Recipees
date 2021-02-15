import json
import numpy as np
from typing import Union
from ase.io import read, write
from asr.core import command, option, DictStr, ASRResult, prepare_result


def get_kpts_size(atoms, density):
    """Try to get a reasonable monkhorst size which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=density)
    size[2] = 1
    for i in range(2):
        if size[i] % 6 != 0:
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts


@command("asr.plot_scs_bs")
@option("--title", type = str, help = "Title for the band structure plot")
def plot_scs_bs(title: str = ""):
    """ Temporary bs visualization tool until we settle on result format """
    import numpy as np
    import matplotlib.pyplot as plt
    from gpaw import GPAW
    from pathlib import Path
    assert Path("bs_scs.gpw").is_file()

    calc = GPAW('bs_scs.gpw', txt=None)
    ef = calc.get_fermi_level()
    bs = calc.band_structure()

    ax = plt.figure(figsize=(14,10)).add_subplot(111)

    xcoords, label_xcoords, orig_labels = bs.get_labels()
    labels = [r'$\Gamma$' if name == 'G' else name for name in orig_labels]
    ax.set_xticks(label_xcoords)
    ax.set_xticklabels(labels, fontsize=40)
    ax.set_ylabel('$E - E_\mathrm{vac}$ [eV]', fontsize=34)
    ax.axis(xmin=0, xmax=xcoords[-1], ymin=-8, ymax=-2)
    for x in label_xcoords[1:-1]:
        ax.axvline(x, color='0.5')
    ax.set_xticks(label_xcoords)
    ax.set_xticklabels(labels, fontsize=40)
    ax.set_ylabel('$E - E_\mathrm{vac}$ [eV]', fontsize=34)
    ax.axis(xmin=0, xmax=xcoords[-1], ymin=-8, ymax=-1.5)

    evac = np.mean(np.mean(calc.get_electrostatic_potential(), axis=0), axis=0)[0]
    ef = ef - evac
    for e in bs.energies[0].T:
        things = ax.plot(xcoords, e - evac, c='b')
    ax.set_title(title, fontsize = 32)
    ax.legend(fontsize = 20)
    ax.xaxis.set_tick_params(width=3, length=15)
    ax.yaxis.set_tick_params(width=3, length=15)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.xticks(size=32)
    plt.yticks(size=32)


    plt.ylim([ef - 5 , ef + 5])
    plt.tight_layout()
    plt.show()





@command(module='asr.scs',
        creates=['gs_scs.gpw'],
        requires=['structure.json'])
@option("--structure", type = str)
@option("--kpts", type = float, help = "In-plane kpoint density")
@option("--calculator", type = DictStr(), help = "Calculator params.")
def calculate_gs(structure: str = "structure.json",
        kpts = 12,
        calculator: dict = {
        'mode': 'lcao',
        'xc': 'PBE',
        'basis': 'dzp',
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'nbands': 'nao',
        'txt': 'gs_scs.txt'}):
    """ Calculates the SCS ground state file

    This recipe calculates and saves the gs_scs.gpw file based on the structure in
    'structure.json' unless another structure file is specified.
    """
    import json
    import numpy as np
    from gpaw import GPAW
    from gpaw.lcao.scissors import Scissors
    from ase.io import read
    atoms = read(structure)
    kpts = get_kpts_size(atoms, kpts)

    shifts = json.load(open("shifts.json", 'r'))
    shift_v1 = shifts['shift_v1']
    shift_c1 = shifts['shift_c1']
    shift_v2 = shifts['shift_v2']
    shift_c2 = shifts['shift_c2']

    tags = atoms.get_tags()
    natoms = len(atoms)
    natoms_l1 = np.extract(tags == 1, tags).shape[0]
    scs = Scissors([(shift_v1, shift_c1, natoms_l1),
                    (shift_v2, shift_c2, natoms - natoms_l1)])

    calculator.update({'eigensolver': scs,
                       'kpts': kpts})

    calc = GPAW(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write('gs_scs.gpw', "all")


@command(module='asr.scs',
         requires=['gs_scs.gpw'],
         creates=['bs_scs.gpw'],
         dependencies=['asr.scs@calculate_gs'])
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints', type=int)
def calculate_bs(kptpath: Union[str, None] = None, npoints: int = 400):
    "Calculate electronic band structure with the self-consistent scissors corrections"
    from gpaw import GPAW
    from gpaw.lcao.scissors import Scissors
    from ase.io import read
    atoms = read('structure.json')
    if kptpath is None:
        path = atoms.cell.bandpath(npoints=npoints, pbc=atoms.pbc, eps=1e-2)
    else:
        path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                                   pbc=atoms.pbc, eps=1e-2)
    parms = {
        'basis': 'dzp',
        'txt': 'bs.txt',
        'fixdensity': True,
        'kpts': path,
        'symmetry': 'off'}
    shifts = json.load(open("shifts.json", 'r'))
    shift_v1 = shifts['shift_v1']
    shift_c1 = shifts['shift_c1']
    shift_v2 = shifts['shift_v2']
    shift_c2 = shifts['shift_c2']

    tags = atoms.get_tags()
    natoms = len(atoms)
    natoms_l1 = np.extract(tags == 1, tags).shape[0]
    scs = Scissors([(shift_v1, shift_c1, natoms_l1),
                    (shift_v2, shift_c2, natoms - natoms_l1)])
    parms.update({'eigensolver': scs})

    calc = GPAW('gs_scs.gpw', **parms)
    calc.get_potential_energy()
    calc.write('bs_scs.gpw', 'all')


