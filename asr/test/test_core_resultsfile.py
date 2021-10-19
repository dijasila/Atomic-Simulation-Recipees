import pytest
from asr.core.resultfile import convert_row_data_to_contexts, make_records_from_contexts
from asr.c2db.relax import Result
from .materials import Ag


@pytest.mark.parametrize("recipename", ["asr.c2db.relax", "asr.c2db.gs:calculate"])
@pytest.mark.parametrize("result", [Result(), {'a': 1}])
@pytest.mark.parametrize("directory", ["some/path/to/a/dir"])  # XXX test if None
@pytest.mark.parametrize(
    "atomic_structures",
    [{"unrelaxed.json": Ag, "structure.json": Ag}],
)
def test_creating_records_from_old_row_data_format(
    recipename,
    result,
    directory,
    atomic_structures,
):
    filename = f'results-{recipename}.json'
    data = {filename: result}
    data.update(atomic_structures)
    contexts = convert_row_data_to_contexts(data, directory)
    context = contexts[0]
    assert context.directory == directory
    assert context.result == result
    assert context.recipename == recipename
    assert context.atomic_structures == atomic_structures

    records = make_records_from_contexts(contexts)
    record = records[0]
    assert record.name == recipename
