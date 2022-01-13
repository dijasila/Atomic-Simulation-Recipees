import pytest
from pytest import approx
import numpy as np


@pytest.mark.ci
def test_stiffness_gpaw(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                        get_webcontent):
    from asr.setup.strains import main as setup_strains
    from asr.stiffness import main as stiffness

    test_material.write('structure.json')
    strain_percent = 1
    setup_strains(strain_percent=strain_percent)

    results = stiffness()
    nd = np.sum(test_material.pbc)

    # check that all keys are in results-asr.stiffness.json:
    keys = ['stiffness_tensor', 'eigenvalues']
    if nd == 2:
        keys.extend(['speed_of_sound_x', 'speed_of_sound_y',
                     'c_11', 'c_22', 'c_33', 'c_23', 'c_13', 'c_12'])
    for key in keys:
        assert key in results

    if nd == 1:
        stiffness_tensor = 0.
        eigenvalues = 0.
    elif nd == 2:
        stiffness_tensor = np.zeros((3, 3))
        eigenvalues = np.zeros(3)
    else:
        stiffness_tensor = np.zeros((6, 6))
        eigenvalues = np.zeros(6)

    assert results['stiffness_tensor'] == approx(stiffness_tensor)
    assert results['eigenvalues'] == approx(eigenvalues)

    content = get_webcontent()
    assert 'Dynamical(stiffness)' in content, content


@pytest.mark.ci
# @pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
#                                   'Pd', 'Pt', 'C'])
@pytest.mark.parametrize('name', ['Al'])
def test_stiffness_emt(asr_tmpdir_w_params, name, get_webcontent):
    from pathlib import Path
    from ase.build import bulk
    from asr.relax import main as relax
    from asr.setup.strains import main as setup_strains
    from asr.setup.params import main as setup_params
    from asr.stiffness import main as stiffness
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)

    structure = bulk(name)
    structure.set_initial_magnetic_moments([0.0] * len(structure))
    structure.write('structure.json')
    strain_percent = 1
    setup_strains(strain_percent=1)

    ij = get_relevant_strains(structure.pbc)
    for i, j in ij:
        for sign in [+1, -1]:
            name = get_strained_folder_name(strain_percent * sign, i, j)
            folder = Path(name)
            assert folder.is_dir()
            # run relaxation in each subfloder with EMT calculator
            from asr.core import chdir
            with chdir(folder):
                import os
                assert os.path.isfile('unrelaxed.json')
                assert os.path.isfile('results-asr.setup.params.json')
                params = {
                    'asr.relax': {'calculator': {'name': 'emt'}}
                }
                setup_params(params=params)
                relax.cli([])
                assert os.path.isfile('results-asr.relax.json')
                assert os.path.isfile('structure.json')

    results = stiffness()

    # check that stiffness_tensor is symmetric
    stiffness_tensor = results['stiffness_tensor']
    assert stiffness_tensor == approx(stiffness_tensor.T, abs=1)

    content = get_webcontent()
    assert 'Dynamical(stiffness)' in content, content
