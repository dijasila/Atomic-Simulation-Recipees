"""Test for anharmonic_phonons recipe."""
import pytest
from pathlib import Path


@pytest.mark.ci
def test_anharmonic_phonons():
    """test for anharmonic_phonons"""
    from asr.anharmonic_phonons import main
    from ase.build import bulk

    atoms = bulk('Ni')
    atoms.write("structure.json")

    main(cellsize=3, calculator={'name': 'lj'}, nd=3, mesh_ph3=1,
         number_structures=1, t1=300, t2=300, cut1=3.5, cut2=2.5, cut3=1.5)

    result = Path('results-asr.anharmonic_phonons_result.json')
    assert result.is_file()
