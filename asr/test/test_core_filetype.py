import pytest
import pathlib

from asr.core.filetype import ExternalFile, ASRFile


@pytest.fixture
def afile(asr_tmpdir):
    filename = 'somefile.txt'
    p = pathlib.Path(filename)
    return p


@pytest.mark.ci
def test_external_file_type_str(afile):
    afile.write_text('abc')
    ext_file = ExternalFile(afile)
    assert str(ext_file) == (
        f'ExternalFile(path={afile.absolute()}, '
        'sha256=ba7816bf8f...)'
    )


@pytest.mark.ci
def test_external_file_type_hash(afile):
    afile.write_text('def')
    ext_file = ExternalFile(afile)
    assert ext_file.sha256 == (
        'cb8379ac2098aa165029e3938a51da'
        '0bcecfc008fd6795f401178647f96c5b34'
    )


@pytest.mark.ci
def test_asr_file_type_hash(asr_tmpdir, afile):
    directory = pathlib.Path('.asr')
    directory.mkdir()
    afile = (directory / afile)
    afile.write_text('def')
    ext_file = ASRFile(afile)
    assert ext_file.sha256 == (
        'cb8379ac2098aa165029e3938a51da'
        '0bcecfc008fd6795f401178647f96c5b34'
    )
