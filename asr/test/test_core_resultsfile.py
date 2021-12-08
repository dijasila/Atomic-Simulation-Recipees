import os
from pathlib import Path

import pytest
from asr.core.resultfile import get_resultfile_records_in_directory


ASR_TEST_DIR = os.environ.get("ASR_TEST_DATA")

if ASR_TEST_DIR is not None:
    directories = (Path(ASR_TEST_DIR) / "EXAMPLE_MATERIAL_FOLDERS").absolute().glob("*")
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
    assert len(resultfile_records)
