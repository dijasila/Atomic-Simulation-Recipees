from ase.calculators.calculator import get_calculator_class
from ase.io import read
import numpy as np 
import os
from pathlib import Path

def calc_vdw(folder):
    Calculator = get_calculator_class("dftd3")
    calc = Calculator()

    atoms = read(f"{folder}/structure.json")
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()
    np.save(f"{folder}/vdw_e.npy", e)

if __name__ == "__main__":
    calc_vdw(Path(".").absolute())
