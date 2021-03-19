import json
import numpy as np
from typing import Union
from ase.io import read, write
from asr.core import command, option, DictStr, ASRResult, prepare_result
from asr.utils.bands import Bands


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
@option("--title", type=str, help="Title for the band structure plot")
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

    ax = plt.figure(figsize=(14, 10)).add_subplot(111)

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
    ax.set_title(title, fontsize=32)
    ax.legend(fontsize=20)
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
@option("--structure", type=str)
@option("--kpts", type=float, help="In-plane kpoint density")
@option("--calculator", type=DictStr(), help="Calculator params.")
def calculate_gs(structure: str = "structure.json",
                 kpts=12,
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
    atoms = read(structure)

    kpts = get_kpts_size(atoms, kpts)
    calculator.update({'kpts': kpts})

    calc = GPAW(**calculator)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write('gs_lcao.gpw', "all")


@command(module='asr.scs',
         requires=['gs_lcao.gpw'],
         creates=['bs_lcao.gpw'],
         dependencies=['asr.scs@calculate_gs'])
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints', type=int)
@option('--eps', type=float, help='Tolerance over symmetry determination')
def calculate_bs(kptpath: Union[str, None] = None, npoints: int = 200, eps: float=1e-1):
    "Calculate electronic band structure with the self-consistent scissors corrections"
    from gpaw import GPAW
    from gpaw.lcao.scissors import Scissors
    from ase.io import read
    atoms = read('structure.json')
    if kptpath is None:
        path = atoms.cell.bandpath(npoints=npoints, pbc=atoms.pbc, eps=eps)
    else:
        path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                                   pbc=atoms.pbc, eps=eps)
    parms = {
        'basis': 'dzp',
        'txt': 'bs.txt',
        'fixdensity': True,
        'kpts': path,
        'symmetry': 'off'}

    calc = GPAW('gs_scs.gpw', **parms)
    calc.get_potential_energy()
    calc.write('bs_lcao.gpw')
    bands = Bands('bs_lcao.gpw')
    bands.dump_to_json()


@command('asr.scs')
@option('--structure', type=str)
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints', type=int)
@option('--kpts', type=float, help="In-plane kpoint density")
@option('--calculator', type=DictStr(), help="Calculator params.")
@option('--gs', type=bool, is_flag=True, help='Request only ground state calculation')
@option('--bs', type=bool, is_flag=True, help='Request only band structure calculation')
@option('--eps', type=float, help='Tolerance over symmetry determination')
def main(structure: str = 'structure.json',
         kptpath: Union[str, None] = None,
         npoints: int = 200,
         kpts: int = 12,
         gs: Union[bool, None] = None,
         bs: Union[bool, None] = None,
         calculator: dict = {
             'mode': 'lcao',
             'xc': 'PBE',
             'basis': 'dzp',
             'occupations': {'name': 'fermi-dirac',
                             'width': 0.05},
             'nbands': 'nao',
             'txt': 'gs_scs.txt'
         },
         eps: float = 1e-1):

    if gs:
        calculate_gs(structure, kpts, calculator)
    if bs:
        calculate_bs(kptpath, npoints, eps)
    if not gs and not bs:
        calculate_gs(structure, kpts, calculator)
        calculate_bs(kptpath, npoints, eps)


if __name__ == '__main__':
    main.cli()
