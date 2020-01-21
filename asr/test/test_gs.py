def test_gs_calculate_gpaw_mock(isolated_filesystem, mock_GPAW):
    from ase.build import bulk
    from asr.gs import calculate
    structure = bulk('C')
    structure.write('structure.json')
    calculate()


def test_gs_gpaw_mock(isolated_filesystem, mock_GPAW):
    from ase.build import bulk
    from asr.gs import main
    structure = bulk('C')
    structure.write('structure.json')
    main()
