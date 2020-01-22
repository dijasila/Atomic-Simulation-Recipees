def test_bandstructure_gpaw_mock(isolated_filesystem, mock_GPAW):
    from ase.build import bulk
    from asr.bandstructure import main
    structure = bulk('C')
    structure.write('structure.json')
    main(skip_deps=True)
