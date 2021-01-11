import pytest

import numpy as np

from asr.core.cache import Cache, FileCacheBackend, MemoryBackend
from asr.core.specification import construct_run_spec
from asr.core.record import RunRecord

from .materials import BN


@pytest.mark.ci
def test_cache(asr_tmpdir):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})

    cache = Cache(backend=FileCacheBackend())

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    assert cache.get(run_specification=run_spec) == run_record
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 2},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    assert cache.get(run_specification=run_spec) == run_record


@pytest.mark.ci
@pytest.mark.parametrize(
    'backend', [FileCacheBackend, MemoryBackend]
)
@pytest.mark.parametrize(
    'obj_to_add',
    [
        1,
        1.02,
        'a',
        (1, 'a'),
        [1, 'a'],
        np.array([1.1, 2.0], float),
        BN,
    ],
)
def test_cache_add(asr_tmpdir, obj_to_add, backend):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(
        run_specification=run_spec,
        result=obj_to_add,
    )
    cache = Cache(backend=backend())

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    fetched_record = cache.get(**run_record)
    if isinstance(obj_to_add, tuple):
        pytest.xfail(reason='JSONSerializer cannot distinguish tuple. '
                     'They are cast to lists.')

    assert run_record == fetched_record
