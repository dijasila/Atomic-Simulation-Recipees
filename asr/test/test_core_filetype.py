import pytest
from pathlib import Path

from asr.core.filetype import ExternalFile


@pytest.fixture
def afile(asr_tmpdir):
    return Path('somefile.txt')


@pytest.mark.ci
def test_external_file_type_str(afile):
    afile.write_text('abc')
    ext_file = ExternalFile(afile.absolute())
    assert ext_file.name == afile.name
    assert str(ext_file) == (
        f'ExternalFile(path={afile.absolute()}, '
        'sha256=ba7816bf8f...)'
    )


@pytest.mark.ci
def test_external_file_type_hash(afile):
    afile.write_text('def')
    ext_file = ExternalFile(afile.absolute())
    assert ext_file.sha256 == (
        'cb8379ac2098aa165029e3938a51da'
        '0bcecfc008fd6795f401178647f96c5b34'
    )
