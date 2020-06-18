import pytest
from asr.core import read_json


@pytest.mark.ci
def test_info(asr_tmpdir):
    from asr.info import main

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
