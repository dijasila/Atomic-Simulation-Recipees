import pytest


@pytest.mark.parametrize('target_magmom', [0, 1, 2])
@pytest.mark.parametrize('magmom', [0, 1, 2])
@pytest.mark.ci
def test_check_magmoms(magmom, target_magmom):
    from asr.zfs import check_magmoms

    try:
        test = check_magmoms(magmom, target_magmom)
        assert test
    except AssertionError:
        assert target_magmom != pytest.approx(magmom)
