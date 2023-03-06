import pytest
from ase.build import bulk
from asr.structureinfo import main


@pytest.mark.ci
def test_cod_id(asr_tmpdir):
    atoms = bulk('Si')
    atoms.write('structure.json')
    result = main()
    assert result['spacegroup'] == 'Fd-3m'


def test_layergroup():
    try:
        from spglib.spglib import get_symmetry_layerdataset
    except ImportError:
        pytest.skip('spglib does not have get_symmetry_layerdataset')

    from asr.structureinfo import get_layer_group
    from ase.build.surface import fcc111

    atoms = fcc111('Au', size=(1, 1, 1), vacuum=4.0)

    lgname, lgnum = get_layer_group(atoms, symprec=0.12345)
    assert lgnum == 80
