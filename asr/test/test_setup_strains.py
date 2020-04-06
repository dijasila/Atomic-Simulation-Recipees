import pytest
from .conftest import test_materials


@pytest.mark.ci
@pytest.mark.parametrize("pbc", [[True, ] * 3,
                                 [True, True, False],
                                 [False, False, True]])
def test_setup_strains_get_relevant_strains(separate_folder, pbc):
    from asr.setup.strains import get_relevant_strains

    ij = set(get_relevant_strains(pbc))
    if sum(pbc) == 3:
        ij2 = {(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
    elif sum(pbc) == 2:
        ij2 = {(0, 0), (1, 1), (0, 1)}
    elif sum(pbc) == 1:
        ij2 = {(2, 2)}

    assert ij == ij2


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_setup_strains(separate_folder, mockgpaw, atoms):
    from asr.setup.strains import (main,
                                   get_strained_folder_name,
                                   get_relevant_strains)
    from asr.core import read_json
    from ase.io import write
    from pathlib import Path
    from asr.setup.params import main as setupparams
    write('structure.json', atoms)
    setupparams(params={'asr.gs@calculate': {
        'calculator': {'mode': 'fd', None: None}}})

    main(strain_percent=1)
    ij = get_relevant_strains(atoms.pbc)
    for i, j in ij:
        name = get_strained_folder_name(1, i, j)
        folder = Path(name)
        assert folder.is_dir()

        paramfile = folder / 'params.json'
        assert paramfile.is_file()

        params = read_json(paramfile)

        assert 'size' in params['asr.relax']['calculator']['kpts']
        assert params['asr.relax']['fixcell']
        assert params['asr.relax']['allow_symmetry_breaking']
        assert params['asr.gs@calculate']['calculator']['mode'] == 'fd'
