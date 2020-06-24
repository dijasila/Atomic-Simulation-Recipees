from asr.dimensionality import main as dimensionality
from asr.dimensionality import get_dimtypes


def test_dimensionality(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)

    results = dimensionality(test_material)

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'
    primary = results['dim_primary']

    dimtypes = get_dimtypes()
    scores = [results[f'dim_score_{dimtype}'] for dimtype in dimtypes]
    assert results[f'dim_score_{primary}'] == max(scores)


def test_dimensionality_cli(asr_tmpdir, test_material):
    nd = sum(test_material.pbc)
    test_material.write('structure.json')
    results = dimensionality.cli(args=[])

    interval = results['k_intervals'][0]
    assert interval['dimtype'] == f'{nd}D'
