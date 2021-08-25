import pytest
from ase import Atoms

from asr.c2db.deformationpotentials import EdgesResult, _main, main
from asr.c2db.relax import Result as RelaxResult


def relax(atoms):
    return RelaxResult.fromdata(atoms=atoms)


def calculate(atoms, vbm_position, cbm_position):
    x = atoms.cell[0, 0] - 1.0
    y = atoms.cell[1, 1] - 1.0
    return EdgesResult.fromdata(
        evbm=-5.0,
        ecbm=-4.0 + x + 2 * y,
        vacuumlevel=1.0)


@pytest.mark.ci
def test_def_pots():
    atoms = Atoms(cell=[1, 1, 1], pbc=[1, 1, 0])
    position = None
    edges, defpots = _main(
        atoms,
        relax_atoms=relax,
        calculate_band_edges=calculate,
        vbm_position=position,
        cbm_position=position)
    assert defpots[:, 0] == pytest.approx(0.0)
    assert defpots[:, 1] == pytest.approx([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.integration_test
@pytest.mark.integration_test_gpaw
def test_def_pots_gpaw_h2(asr_tmpdir):
    d = 2.0
    atoms = Atoms('HH',
                  positions=[(0, 0, 0), (0, 0, 0.75)],
                  cell=[d, d, 5],
                  pbc=[1, 1, 0])
    atoms.center(axis=2)
    params = dict(name='gpaw',
                  mode=dict(name='pw', ecut=300),
                  kpts=dict(density=1.0))
    result = main(atoms, params)
    defpots = result['deformation_potentials']
    assert defpots[:, 0] == pytest.approx(
        [-6.3, -6.3, 0.0, 0.0, 0.0, -2.8], abs=0.1)
    assert defpots[:, 1] == pytest.approx(
        [1.7, 1.7, 0.0, 0.0, 0.0, -1.1], abs=0.1)
