import pytest
import pathlib

from asr.core.filetype import ExternalFile, ASRPath


@pytest.fixture
def afile(asr_tmpdir):
    filename = 'somefile.txt'
    p = pathlib.Path(filename)
    return p


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


@pytest.fixture
def asr_file_path(asr_tmpdir):
    filename = 'filename.txt'
    path = pathlib.Path(filename)
    return path, filename


@pytest.mark.ci
def test_asr_file_type(asr_file_path):
    path, filename = asr_file_path

    asr_path = ASRPath(path)

    assert str(asr_path) == str(pathlib.Path('.asr').absolute() / filename)


@pytest.mark.ci
def test_asr_file_type_repr(asr_file_path):
    path, filename = asr_file_path
    asr_path = ASRPath(path)
    assert str(asr_path) == repr(asr_path)


@pytest.fixture
def another_asr_file_path(asr_tmpdir):
    filename = 'filename2.txt'
    path = pathlib.Path(filename)
    return path, filename


@pytest.mark.ci
def test_asr_file_type_eq_not_true(asr_file_path, another_asr_file_path):
    path, _ = asr_file_path
    path2, _ = another_asr_file_path

    asr_path = ASRPath(path)
    another_asr_path = ASRPath(path2)
    assert not asr_path == another_asr_path


@pytest.mark.ci
def test_asr_file_type_eq_is_true(asr_file_path):
    path, _ = asr_file_path

    asr_path = ASRPath(path)
    another_asr_path = ASRPath(path)
    assert asr_path == another_asr_path
