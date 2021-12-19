import os
from pathlib import Path

import pytest
from asr.core.resultfile import get_resultfile_records_in_directory


ASR_TEST_DIR = os.environ.get("ASR_TEST_DATA")

if ASR_TEST_DIR is not None:
    directories = list(
        (Path(ASR_TEST_DIR) / "EXAMPLE_MATERIAL_FOLDERS").absolute().glob("*")
    )
else:
    directories = []


@pytest.mark.skipif(
    ASR_TEST_DIR is None,
    reason="No databases to test migration on.",
)
@pytest.mark.parametrize(
    "directory", directories, ids=[str(pth) for pth in directories]
)
@pytest.mark.integration_test_gpaw
# Needs gpaw to extract parameters from existing .gpw files
def test_migrating_directory_succeeds(asr_tmpdir, directory):
    resultfile_records = get_resultfile_records_in_directory(directory)
    assert resultfile_records
