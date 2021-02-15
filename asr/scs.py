import numpy as np
import json
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





@command("asr.scs@calculate_gs")
@option("--structure", type = str)
@option("--shifts_file", type = str)
@option("--kpts", type = float, help = "In-plane kpoint density")
@option("--calculator", type = DictStr(), help = "Calculator params.")
def calculate_gs(structure: str = "structure.json",
        shifts_file: str = "shifts.json",
        kpts = 10,
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

    shifts = json.load(open(shifts_file, 'r'))
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
    atoms.calc.write('gs.gpw')

