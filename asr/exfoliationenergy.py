from asr.core import command, option, AtomsFile, read_json, ASRResult, prepare_result
from ase.calculators.calculator import get_calculator_class
from pathlib import Path
from ase import Atoms
from typing import Dict


@prepare_result
class ExfoliationResult(ASRResult):
    exfoliationenergy: float
    most_stable_bilayer: str
    number_of_bilayers: int
    bilayer_names: Dict[str, float]

    key_descriptions = dict(
        exfoliationenergy='Estimated exfoliation energy',
        most_stable_bilayer='uid of most stable bilayer',
        number_of_bilayers='The number',
        bilayer_names='uids of bilayers')


def get_bilayers_energies(p):
    bilayers_energies = {}
    for sp in p.iterdir():
        if not sp.is_dir():
            continue
        datap = sp / 'results-asr.relax_bilayer.json'
        if datap.exists():
            data = read_json(str(datap))
            bilayers_energies[str(sp)] = data['energy']

    return bilayers_energies


def monolayer_energy(atoms):
    return atoms.get_potential_energy()


def vdw_energy(atoms, path):
    precalculated_file = str(path.resolve()) + "/vdw_e.npy"
    if Path(precalculated_file).is_file():
        import numpy as np
        vdw_e = np.load(precalculated_file).item()
        return vdw_e
    atoms = atoms.copy()
    Calculator = get_calculator_class('dftd3')
    calc = Calculator()
    atoms.calc = calc
    vdw_e = atoms.get_potential_energy()
    return vdw_e


def calculate_exfoliation(ml_e, vdw_e, bilayers_energies, atoms):
    import numpy as np
    most_stable, bi_e = min(bilayers_energies, key=lambda t: t[1])

    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))

    exf_e = (2 * (ml_e + vdw_e) - bi_e) / area
    return exf_e, most_stable, [f for f, s in bilayers_energies]


@command('asr.exfoliationenergy')
@option('-a', '--atoms', help='Monolayer file',
        type=AtomsFile(), default='./structure.json')
def main(atoms: Atoms) -> ASRResult:
    """Calculate monolayer exfoliation energy.

    Assumes there are subfolders containing relaxed bilayers,
    and uses these to estimate the exfoliation energy.
    """
    p = Path('.')
    bilayers_energies = get_bilayers_energies(p)

    if len(bilayers_energies) == 0:
        return ExfoliationResult.fromdata(exfoliationenergy=None,
                                          most_stable_bilayer=None,
                                          number_of_bilayers=0,
                                          bilayer_names=[])
        # raise ValueError('No bilayers found!')

    ml_e = monolayer_energy(atoms)
    vdw_e = vdw_energy(atoms, p)

    things = calculate_exfoliation(ml_e, vdw_e,
                                   bilayers_energies, atoms)
    exf_energy, most_stable, bilayer_names = things

    results = ExfoliationResult.fromdata(exfoliationenergy=exf_energy,
                                         most_stable_bilayer=most_stable,
                                         number_of_bilayers=len(bilayer_names),
                                         bilayer_names=bilayer_names)
    return results
