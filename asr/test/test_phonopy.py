import pytest
from pathlib import Path


@pytest.mark.ci
def test_phonopy(asr_tmpdir_w_params, mockgpaw, get_webcontent):
    """Simple test of phononpy recipe."""
    from asr.phonopy import calculate, main
    from asr.core import read_json
    from ase.build import bulk

    N = 2

    atoms = bulk('Al', 'fcc', a=4.05)
    atoms.write("structure.json")

    calculate(supercell=[N, N, N], calculator={'name': 'emt'})

    main()

    result = Path('results-asr.phonopy.json')
    assert result.is_file()
    data = read_json('results-asr.phonopy.json')
    assert data['minhessianeig'] == pytest.approx(0)
    assert data['dynamic_stability_level'] == 3
