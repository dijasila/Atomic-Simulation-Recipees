import pytest
from .materials import std_test_materials


@pytest.mark.parametrize('reduced', [True, False])
@pytest.mark.ci
def test_get_stoichiometry(reduced):
    from asr.charge_neutrality import get_stoichiometry

    materials = std_test_materials.copy()
    results = {'Si2': {'Si': 2},
               'BN': {'B': 1, 'N': 1},
               'Ag': {'Ag': 1},
               'Fe': {'Fe': 1}}
    for atom in materials:
        stoi = get_stoichiometry(atom, reduced=reduced)
        for el in stoi:
            ref = results[f'{atom.get_chemical_formula()}']
            if reduced and atom.get_chemical_formula() == 'Si2':
                assert stoi[el] == 1
            else:
                assert ref[el] == stoi[el]


@pytest.mark.parametrize('atoms', std_test_materials.copy())
@pytest.mark.ci
def test_get_element_list(atoms):
    from asr.charge_neutrality import get_element_list

    element_list = get_element_list(atoms)
    for i in range(len(atoms)):
        if atoms.get_chemical_formula() == 'Si2':
            assert atoms.symbols[i] == element_list[0]
            assert len(element_list) == 1
        else:
            assert atoms.symbols[i] == element_list[i]
            assert len(element_list) == len(atoms)


@pytest.mark.parametrize('hof', [-1, -2])
@pytest.mark.ci
def test_get_adjusted_chemical_potentials(hof):
    from asr.charge_neutrality import (get_adjusted_chemical_potentials,
                                       get_element_list)
    from .materials import BN

    atoms = BN.copy()
    element_list = get_element_list(atoms)

    for element in element_list:
        chempot = get_adjusted_chemical_potentials(atoms, hof, element)
        for symbol in chempot:
            if symbol == element:
                assert chempot[f'{symbol}'] == pytest.approx(hof)
            else:
                assert chempot[f'{symbol}'] == pytest.approx(0)
