import pytest
import numpy as np
from ase import Atoms


@pytest.mark.ci
def test_formalpolarization(asr_tmpdir_w_params, mockgpaw, test_material):
    from asr.formalpolarization import main
    main(atoms=test_material)


abn = 2.51
BN = Atoms(
    "BN",
    scaled_positions=[[0, 0, 0], [1 / 3, 2 / 3, 0]],
    cell=[
        [abn, 0.0, 0.0],
        [-0.5 * abn, np.sqrt(3) / 2 * abn, 0],
        [0.0, 0.0, 15.0],
    ],
    pbc=[True, True, False],
)

Ag_chain = Atoms(
    "Ag",
    scaled_positions=[[0, 0, 0]],
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 2],
    ],
    pbc=[False, False, True],
)

bad_atoms = [BN, Ag_chain]


@pytest.mark.ci
@pytest.mark.parametrize("atoms", bad_atoms)
def test_formalpolarization_test_atoms_too_close_to_boundary(
        asr_tmpdir_w_params, mockgpaw, atoms):
    from asr.formalpolarization import AtomsTooCloseToBoundary, main

    with pytest.raises(AtomsTooCloseToBoundary):
        main(atoms=atoms)
