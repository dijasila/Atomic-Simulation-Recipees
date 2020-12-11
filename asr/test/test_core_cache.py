from asr.core.cache import Cache, FileCacheBackend
from asr.core.specification import construct_run_spec
from asr.core.record import RunRecord


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
