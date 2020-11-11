from .materials import std_test_materials
import pytest


@pytest.mark.ci
def test_asr_cli_results_figures_gs():
    from asr.core.material import Material
    atoms = std_test_materials[0]
    kvp = {'gap': 1}
    data = {}

    material = Material(atoms, kvp, data)
    assert "gap" in material

    keys = [key for key in material]
    assert "gap" in keys


@pytest.mark.ci
def test_material_count_atoms_equals_len_atoms(test_material):
    from asr.core.material import Material
    kvp = {}
    data = {}

    material = Material(test_material, kvp, data)
    count = {}
    for symbol in test_material.symbols:
        count[symbol] = count.get(symbol, 0) + 1
    assert material.count_atoms() == count
