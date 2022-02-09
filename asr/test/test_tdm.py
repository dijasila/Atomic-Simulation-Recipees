import pytest
from pathlib import Path
from asr.defect_symmetry import WFCubeFile
from asr.tdm import (get_state_numbers_from_defectpath,
                     get_center_of_cell)
from .materials import std_test_materials


@pytest.mark.parametrize('minband', [0, 1, 2])
@pytest.mark.parametrize('maxband', [10, 11, 12])
@pytest.mark.ci
def test_get_state_numbers_defect(asr_tmpdir, minband, maxband):
    for band in [minband, 8, 9, maxband]:
        for spin in [0, 1]:
            wfcubefile = WFCubeFile(band=band, spin=spin)
            filename = wfcubefile.filename
            Path(filename).touch()

    defectpath = Path('.')
    n1, n2 = get_state_numbers_from_defectpath(defectpath)
    assert n1 == pytest.approx(minband)
    assert n2 == pytest.approx(maxband)


@pytest.mark.ci
def test_get_center_of_cell():
    structures = std_test_materials.copy()
    for atoms in structures:
        cell = atoms.get_cell()
        center = get_center_of_cell(atoms)
        for i in range(3):
            assert center[i] == pytest.approx(0.5 * sum(cell[:, i]))
