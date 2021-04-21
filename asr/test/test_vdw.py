import pytest


@pytest.mark.integration_test
def test_vdw_energy(asr_tmpdir, test_material):
    from asr.vdw import main
    results = main(test_material)
    assert results['vdw_energy'] == pytest.approx(1e-3)
