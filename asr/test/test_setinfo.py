"""Tests of the asr.setinfo module."""
import pytest
from asr.core import read_json
from asr.setinfo import main


@pytest.mark.ci
def test_info(asr_tmpdir):
    """Test that arguments are correctly overwritten."""
    main([
        ('first_class_material', True),
    ])
    info = read_json('info.json')
    assert info == {'first_class_material': True}

    main([
        ('first_class_material', False),
    ])
    info = read_json('info.json')
    assert info == {'first_class_material': False}

    main([
        ('class', 'TMD'),
    ])
    info = read_json('info.json')
    assert info == {'class': 'TMD',
                    'first_class_material': False}

    main([
        ('class', ''),
    ])
    info = read_json('info.json')
    assert info == {'first_class_material': False}


@pytest.mark.ci
def test_info_raises_with_protected_key(asr_tmpdir):
    """Test that protected keys cannot be arbitrarily set."""
    with pytest.raises(ValueError):
        main([
            ('first_class_material', 'bad key'),
        ])
