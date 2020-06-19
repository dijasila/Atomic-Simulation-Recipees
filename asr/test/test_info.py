import pytest
from asr.core import read_json
from asr.info import main


@pytest.mark.ci
def test_info(asr_tmpdir):
    """Test that arguments are correctly overwritten."""
    main([
        ('primary', True),
    ])
    info = read_json('info.json')
    assert info == {'primary': True}

    main([
        ('primary', False),
    ])
    info = read_json('info.json')
    assert info == {'primary': False}

    main([
        ('class', 'TMD'),
    ])
    info = read_json('info.json')
    assert info == {'class': 'TMD',
                    'primary': False}


@pytest.mark.ci
def test_info_call_from_cli(asr_tmpdir):
    """Test that CLI arguments are handled correctly."""
    main.cli(['primary:True', 'class:"TMD"'])
    info = read_json('info.json')
    assert info == {'primary': True,
                    'class': 'TMD'}


@pytest.mark.ci
def test_info_raises_with_protected_key(asr_tmpdir):
    """Test that protected keys cannot be arbitrarily set."""
    with pytest.raises(ValueError):
        main([
            ('primary', 'bad key'),
        ])
