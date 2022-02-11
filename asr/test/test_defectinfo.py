import pytest
from .materials import BN, Ag
from asr.defectinfo import get_nearest_neighbor_distance


@pytest.mark.ci
def test_get_nearest_neighbor_distance():
    materials = [BN.copy(), Ag.copy()]
    for atoms in materials:
        cell = atoms.get_cell()
        lengths = cell.lengths()
        distance = get_nearest_neighbor_distance(atoms)

        assert distance == pytest.approx(min(lengths))
