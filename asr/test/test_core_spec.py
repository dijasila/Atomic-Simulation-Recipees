import pytest
from asr.core.specification import construct_run_spec

from .materials import BN


@pytest.mark.ci
@pytest.mark.parametrize(
    'parameters',
    [
        dict(atoms=BN),
        dict(atoms=BN, calculator=dict(name='gpaw', mode='fd'))
    ]
)
def test_spec_equal(parameters):
    spec = construct_run_spec(
        name='test',
        parameters=parameters,
    )

    assert spec == spec
