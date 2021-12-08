import os
import shutil
from pathlib import Path

import pytest

from asr.c2db.relax import Result
from asr.core.resultfile import (convert_row_data_to_contexts,
                                 make_records_from_contexts)

from .materials import Ag
from asr.core.resultfile import get_resultfile_records_in_directory


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


ASR_TEST_DIR = os.environ.get("ASR_TEST_DATA")

if ASR_TEST_DIR is not None:
    directories = (
        Path(ASR_TEST_DIR) / "EXAMPLE_MATERIAL_FOLDERS"
    ).absolute().glob("*")
else:
    directories = []

@pytest.mark.skipif(
    ASR_TEST_DIR is None,
    reason="No databases to test migration on.",
)
@pytest.mark.parametrize("directory", directories)
@pytest.mark.ci
def test_migrating_realistic_directories(asr_tmpdir, directory):
    resultfile_records = get_resultfile_records_in_directory(directory)
    assert resultfile_records
