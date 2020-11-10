from asr.core import command, option, AtomsFile, read_json
from ase.calculators.calculator import get_calculator_class
from pathlib import Path
from ase import Atoms


class ExfoliationResults:
    def __init__(self, exf_energy, most_stable_bilayer,
                 bilayer_names):
        self.exfoliationenergy = exf_energy
        self.most_stable_bilayer = most_stable_bilayer
        self.number_of_bilayers = len(bilayer_names)
        self.bilayer_names = bilayer_names

    def to_dict(self):
        attrs = [x for x in dir(self) if "__" not in x and not callable(getattr(self, x))]
        return {x: getattr(self, x)
                for x in attrs}

    def default():
        return ExfoliationResults("No result", "None", [])


def get_bilayers_energies(p):
    bilayers_energies = []
    for sp in p.iterdir():
        if not sp.is_dir():
            continue
        datap = Path(str(sp) + "/results-asr.relax_bilayer.json")
        if datap.exists():
            data = read_json(str(datap))
            bilayers_energies.append((str(sp), data['energy']))

    return bilayers_energies


def monolayer_energy(atoms):
    return atoms.get_potential_energy()


def vdw_energy(atoms):
    atoms = atoms.copy()
    Calculator = get_calculator_class('dftd3')
    calc = Calculator()
    atoms.calc = calc
    vdw_e = atoms.get_potential_energy()
    return vdw_e

def calculate_exfoliation(ml_e, vdw_e, bilayers_energies):
    most_stable, bi_e = min(bilayers_energies, key=lambda t: t[1])

    exf_e = 2 * (ml_e + vdw_e) - bi_e
    return exf_e, most_stable, [f for f, s in bilayers_energies]


@command('asr.exfoliationenergy')
@option('-a', '--atoms', help='Monolayer file',
        type=AtomsFile(), default='./structure.json')
def main(atoms: Atoms):
    """Calculate monolayer exfoliation energy.

    Assumes there are subfolders containing relaxed bilayers,
    and uses these to estimate the exfoliation energy.
    """
    p = Path('.')
    bilayers_energies = get_bilayers_energies(p)

    if len(bilayers_energies) == 0:
        return ExfoliationResults.default().to_dict()
        
    ml_e = monolayer_energy(atoms)
    vdw_e = vdw_energy(atoms)

    exf_energy, most_stable, bilayer_names = calculate_exfoliation(ml_e, vdw_e, bilayers_energies)

    results = ExfoliationResults(exf_energy, most_stable, bilayer_names).to_dict()
    print(results)
    return results
