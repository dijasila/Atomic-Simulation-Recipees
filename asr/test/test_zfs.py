import pytest
from asr.zfs import check_magmoms


@pytest.mark.parametrize('target_magmom', [0, 1, 2])
@pytest.mark.parametrize('magmom', [0, 1, 2])
@pytest.mark.ci
def test_check_magmoms(magmom, target_magmom):
    if magmom == target_magmom:
        check_magmoms(magmom, target_magmom)
    else:
        with pytest.raises(ValueError):
            check_magmoms(magmom, target_magmom)
