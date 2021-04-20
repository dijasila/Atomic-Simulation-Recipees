import pytest
from ase.build import bulk

Si = bulk('Si')


@pytest.mark.ci
def test_setup_decorate_si(asr_tmpdir_w_params, mockgpaw):
    from asr.setup.decorate import main
    decorated = main(atoms=Si)

    assert len(decorated) == 2
    assert all(decorated[1]['atoms'].symbols == 'Ge')
