from asr.core.command import FullFeatureFileCache, construct_run_spec, RunRecord


def test_cache(asr_tmpdir):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})
    cache = FullFeatureFileCache()

    assert not cache.has(run_spec)
    cache.add(run_record)
    assert cache.has(run_spec)
    assert cache.get(run_spec) == run_record
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 2},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})

    assert not cache.has(run_spec)
    cache.add(run_record)
    assert cache.has(run_spec)
    assert cache.get(run_spec) == run_record
