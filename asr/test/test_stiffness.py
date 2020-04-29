import pytest
from pytest import approx
import numpy as np


def test_stiffness_gpaw_1(asr_tmpdir_w_params, mockgpaw, mocker, test_material):
    from pathlib import Path
    from asr.relax import main as relax
    from asr.setup.strains import main as setup_strains
    from asr.stiffness import main as stiffness
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)

    test_material.write('structure.json')
    strain_percent = 1
    setup_strains(strain_percent=strain_percent)

    ij = get_relevant_strains(test_material.pbc)
    for i, j in ij:
        for sign in [+1, -1]:
            name = get_strained_folder_name(strain_percent * sign, i, j)
            folder = Path(name)
            assert folder.is_dir()
            # run relaxation in each subfloder with gpaw calculator
            from asr.core import chdir
            with chdir(folder):
                import os
                assert os.path.isfile('unrelaxed.json')
                assert os.path.isfile('results-asr.setup.params.json')
                relax(calculator={"name": "gpaw",
                                  "kpts": {"density": 2, "gamma": True}})
                assert os.path.isfile('results-asr.relax.json')
                assert os.path.isfile('structure.json')

    results = stiffness()

    nd = np.sum(test_material.pbc)
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


def test_stiffness_gpaw_2(asr_tmpdir_w_params, mockgpaw, mocker, test_material):
    from pathlib import Path
    from asr.relax import main as relax
    from asr.setup.strains import main as setup_strains
    from asr.stiffness import main as stiffness
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)

    test_material.write('structure.json')
    strain_percent = 1
    setup_strains(strain_percent=strain_percent)
    nd = np.sum(test_material.pbc)

    ij = get_relevant_strains(test_material.pbc)
    for i, j in ij:
        for sign in [+1, -1]:
            name = get_strained_folder_name(strain_percent * sign, i, j)
            folder = Path(name)
            assert folder.is_dir()
            # run relaxation in each subfloder with gpaw calculator
            from asr.core import chdir
            with chdir(folder):
                from gpaw import GPAW

                # generate random stress (without breaking symmetry)
                from ase.units import J
                stress = np.random.rand(6) * J / 1e30  # does this make any sense?
                if nd == 2:
                    stress *= 1e10
                elif nd == 1:
                    stress *= 1e20
                else:
                    pass

                mocker.patch.object(GPAW, "_get_stress")
                GPAW._get_stress.return_value = stress
                relax(calculator={"name": "gpaw",
                                  "kpts": {"density": 2, "gamma": True}})

    results = stiffness()

    # check that all keys are in results-asr.stiffness.json:
    keys = ['stiffness_tensor', 'eigenvalues']
    if nd == 2:
        keys.extend(['speed_of_sound_x', 'speed_of_sound_y',
                     'c_11', 'c_22', 'c_33', 'c_23', 'c_13', 'c_12'])
    for key in keys:
        assert key in results

    # check that num. of eigenvalues is correct
    eigenvalues = results['eigenvalues']
    if nd == 3:
        assert len(eigenvalues) == 6
    elif nd == 2:
        assert len(eigenvalues) == 3
    # else:
    #     assert len(eigenvalues) == 1


# @pytest.mark.ci
@pytest.mark.parametrize('name', ['Al', 'Cu', 'Ag', 'Au', 'Ni',
                                  'Pd', 'Pt', 'C'])
def test_stiffness_emt(asr_tmpdir_w_params, name):
    from pathlib import Path
    from ase.build import bulk
    from asr.relax import main as relax
    from asr.setup.strains import main as setup_strains
    from asr.setup.params import main as setup_params
    from asr.stiffness import main as stiffness
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)

    structure = bulk(name)
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
                # # should I run relaxation or just mock
                # # an EMT calculation with a well defined stress?
                params = {
                    'asr.relax': {'calculator': {'name': 'emt'}}
                }
                setup_params(params=params)
                relax()
                assert os.path.isfile('results-asr.relax.json')
                assert os.path.isfile('structure.json')

    results = stiffness()

    # check that stiffness_tensor is symmetric
    stiffness_tensor = results['stiffness_tensor']
    s_max = np.max(stiffness_tensor)
    assert np.allclose(stiffness_tensor, stiffness_tensor.T,
                       atol=1e-05 * s_max)
