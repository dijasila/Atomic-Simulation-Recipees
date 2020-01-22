def test_gs_calculate(isolated_filesystem, mock_GPAW):
    from ase.build import bulk
    from asr.gs import calculate
    structure = bulk('C')
    structure.write('structure.json')
    calculate()


def test_gs_main(isolated_filesystem, mock_GPAW):
    from asr.gs import main
    results = main()
    assert results.get('gap') < 0.001
