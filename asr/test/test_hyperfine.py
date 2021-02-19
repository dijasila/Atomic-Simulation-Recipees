import pytest
from pytest import approx
from .materials import BN


@pytest.mark.ci
def test_hyperfine(asr_tmpdir):
    from ase.io import write
    from asr.gs import calculate, main
    from asr.hyperfine import main as hyperfine

    write('structure.json', BN)
    calculate(
        calculator={
            "name": "gpaw",
            "mode": {"name": "pw", "ecut": 500},
            "xc": "PBE",
            "basis": "dzp",
            "kpts": {"density": 2, "gamma": True},
        },
    )
    main()

    res = hyperfine()
    # test HF values
    kinds = ['B', 'N']
    for i, element in enumerate(res['hyperfine']):
        assert element['kind'] == kinds[i]
        assert element['index'] == i
    # test gyromagnetic values
    symbols = ['B', 'N']
    gs = [1.0, 0.40366839]
    for i, element in enumerate(res['gfactors']):
        assert element['symbol'] == symbols[i]
        assert element['g'] == approx(gs[i])
