import pytest
from .materials import std_test_materials


@pytest.mark.ci
def test_get_stoichiometry():
    from asr.charge_neutrality import get_stoichiometry

    materials = std_test_materials.copy()
    results = {'Si2': {'Si': 2},
               'BN': {'B': 1, 'N': 1},
               'Ag': {'Ag': 1},
               'Fe': {'Fe': 1}}
    for atom in materials:
        stoi = get_stoichiometry(atom)
        for el in stoi:
            ref = results[f'{atom.get_chemical_formula()}']
            assert ref[el] == stoi[el]
