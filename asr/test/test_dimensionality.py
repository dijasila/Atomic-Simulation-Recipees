from asr.dimensionality import main as dimensionality
from asr.extra_fluff import get_dimtypes


def test_dimensionality(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)

    results = dimensionality(test_material)

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'
    primary = results['dim_primary']

    dimtypes = get_dimtypes()
    scores = [results[f'dim_score_{dimtype}'] for dimtype in dimtypes]
    assert results[f'dim_score_{primary}'] == max(scores)

    # 1.3 is reasonable. Most physical materials lie around 1
    assert results[f'dim_threshold_{nd}D'] < 1.3
    # These keys should not be contained in results
    for i in range(nd + 1, 4):
        assert f'dim_threshold_{i}D' not in results


def test_dimensionality_cli(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)
    test_material.write('structure.json')
    results = dimensionality.cli(args=[])

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'
