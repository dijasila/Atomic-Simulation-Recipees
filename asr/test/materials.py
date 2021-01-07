"""Module containing a set of test materials.

Important module variables:

  - ``std_test_materials``: List of materials that represent the
    minimal set of materials you should test your recipe on.
  - ``all_test_materials``: List of all test materials in this module.
"""

from ase import Atoms
from ase.build import bulk
import numpy as np


# Make some 1D, 2D and 3D test materials
Si = bulk("Si")
Ag = bulk("Ag")
Ag2 = bulk("Ag").repeat((2, 1, 1))
Fe = bulk("Fe")
Fe.set_initial_magnetic_moments([1])
abn = 2.51
BN = Atoms(
    "BN",
    scaled_positions=[[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]],
    cell=[
        [abn, 0.0, 0.0],
        [-0.5 * abn, np.sqrt(3) / 2 * abn, 0],
        [0.0, 0.0, 15.0],
    ],
    pbc=[True, True, False],
)
Agchain = Atoms(
    "Ag",
    scaled_positions=[[0.5, 0.5, 0]],
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 2],
    ],
    pbc=[False, False, True],
)
GaAs = Atoms(
    'GaAs',
    cell=bulk('Ga', 'fcc', a=5.68).cell, pbc=True,
    scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

std_test_materials = [Si, BN, Agchain, Fe]
all_test_materials = [Si, BN, Agchain, Fe]
