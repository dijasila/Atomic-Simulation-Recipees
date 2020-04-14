import pytest
import numpy as np


@pytest.mark.ci
def test_setup_displacements_displace_atom(test_material):
    from asr.setup.displacements import displace_atom
    delta = 0.02
    pos_av = test_material.get_positions()
    for ia in range(len(test_material)):
        for iv in [0, 1, 2]:
            for sign in [-1, 1]:
                new_atoms = displace_atom(test_material, ia, iv, sign, delta)
                new_pos_av = new_atoms.get_positions()

                posav = pos_av[ia, iv]
                new_posav = new_pos_av[ia, iv]
                assert np.abs(posav - new_posav) == pytest.approx(delta)


@pytest.mark.ci
def test_setup_displacements_get_all_displacements(test_material):
    from asr.setup.displacements import get_all_displacements
    displacements = list(get_all_displacements(test_material))

    test_disps = [(ia, iv, sign) for ia in range(len(test_material)) for iv in
                  range(3) for sign in [-1, 1]]

    for disp in test_disps:
        assert disp in displacements

    assert len(displacements) == len(test_disps)


@pytest.mark.ci
def test_setup_displacements(separate_folder, test_material):
    from asr.setup.displacements import main as displacements

    test_material.write('structure.json')
    results = displacements()
    generated_folders = results['folders']
    assert 'displacements/0.01-0-+x' in generated_folders
