import pytest
from pytest import approx


@pytest.mark.parametrize("efermi", [0, 0.25, 0.99])
@pytest.mark.parametrize("gap", [1, 2])
def test_gs_main(isolated_filesystem, mock_GPAW, gap, efermi):
    mock_GPAW.set_property(gap=gap, fermi_level=efermi)

    from asr.gs import calculate, main

    calculate(calculator={'name': 'gpaw',
                          'mode': {'name': 'pw', 'ecut': 800},
                          'xc': 'PBE',
                          'basis': 'dzp',
                          'kpts': {'density': 2, 'gamma': True},
                          'occupations': {'name': 'fermi-dirac',
                                          'width': 0.05},
                          'convergence': {'bands': 'CBM+3.0'},
                          'nbands': '200%',
                          'txt': 'gs.txt',
                          'charge': 0})

    results = main()
    if gap / 2 > efermi:
        assert results.get('gap') == approx(gap)
    else:
        assert results.get('gap') == approx(0)
    assert results.get('gaps_nosoc').get('efermi') == approx(efermi)
    assert results.get('efermi') == approx(efermi)
