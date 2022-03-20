import pytest
from pathlib import Path


@pytest.mark.ci
def test_anharmonic_phonons():
    """test for anharmonic_phonons"""
    from asr.anharmonic_phonons import  main
    from asr.core import read_json
    from ase.build import bulk

    atoms = bulk('Ni')
    atoms.write("structure.json")

    main(cellsize=5, calculator = 'EMT', nat_dim =3, mesh_ph3 = 1, number_structures=1, t1=300, t2 = 300, cut1=4.0, cut2=3.0, cut3=2.0)

    result = Path('results-asr.anharmonic_phonons3_result.json')
    assert result.is_file()
