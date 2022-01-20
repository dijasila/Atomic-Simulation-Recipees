from .materials import Fe
import pytest
from pytest import approx


# XXX How to test for shifted origin, happens for Si
@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Fe])
def test_setup_symmetrize(asr_tmpdir_w_params, inputatoms):
    import numpy as np
    from asr.setup.symmetrize import main
    rng = np.random.RandomState(1234)
    atoms = inputatoms.copy()
    atoms.rattle(stdev=0.001)
    cell_cv = atoms.get_cell()
    cell_cv += (rng.random((3, 3)) - 0.5) * 1e-5
    atoms.set_cell(cell_cv)

    symmetrizedatoms = main(atoms=atoms)

    assert symmetrizedatoms.cell.cellpar() == approx(inputatoms.cell.cellpar())
    assert (symmetrizedatoms.get_scaled_positions()
            == approx(inputatoms.get_scaled_positions()))
    assert (symmetrizedatoms.get_initial_magnetic_moments()
            == approx(inputatoms.get_initial_magnetic_moments()))
