import pytest
from asr.core import read_json
from asr.info import main


@pytest.mark.ci
def test_info(asr_tmpdir):
    """Test that arguments are correctly overwritten."""
    main([
        ('material_type', 'primary'),
    ])
    info = read_json('info.json')
    assert info == {'material_type': 'primary'}

    main([
        ('material_type', 'secondary'),
    ])
    info = read_json('info.json')
    assert info == {'material_type': 'secondary'}

    main([
        ('class', 'TMD'),
    ])
    info = read_json('info.json')
    assert info == {'class': 'TMD',
                    'material_type': 'secondary'}


@pytest.mark.ci
def test_info_call_from_cli(asr_tmpdir):
    """Test that CLI arguments are handled correctly."""
    main.cli(['material_type:primary', 'class:TMD'])
    info = read_json('info.json')
    assert info == {'material_type': 'primary',
                    'class': 'TMD'}


@pytest.mark.ci
def test_info_raises_with_protected_key(asr_tmpdir):
    """Test that protected keys cannot be arbitrarily set."""
    with pytest.raises(ValueError):
        main([
            ('material_type', 'bad key'),
        ])
