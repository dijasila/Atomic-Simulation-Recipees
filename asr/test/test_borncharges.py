import pytest
from .conftest import test_materials
import numpy as np


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_borncharges(separate_folder, mockgpaw, atoms):
    from asr.borncharges import main
    atoms.write('structure.json')
    results = main()

    Z_avv = results['Z_avv']
    for Z_vv in Z_avv:
        assert np.allclose(Z_vv, np.eye(3))
