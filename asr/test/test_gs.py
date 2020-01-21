def test_gs_gpaw_mock(isolated_filesystem, mock_GPAW):
    from ase.build import bulk
    from asr.gs import main
    structure = bulk('C')
    structure.write('structure.json')
    results = main()
    assert results.get('gap') < 0.001
