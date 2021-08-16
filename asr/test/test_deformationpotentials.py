from ase import Atoms
from asr.deformationpotentials import _main, EdgesResult
from asr.relax import Result as RelaxResult


def relax(atoms):
    return RelaxResult.fromdata(atoms=atoms)


def calculate(atoms, vbm_position, cbm_position):
    x = atoms.cell[0, 0] - 1.0
    y = atoms.cell[1, 1] - 1.0
    return EdgesResult.fromdata(
        evbm=-5.0,
        ecbm=-4.0 + x + 2 * y,
        vacuumlevel=1.0)


def test_def_pots():
    atoms = Atoms(cell=[1, 1, 1], pbc=[1, 1, 0])
    position = None
    edges, defpots = _main(
        atoms,
        relax_atoms=relax,
        calculate_band_edges=calculate,
        vbm_position=position,
        cbm_position=position)
