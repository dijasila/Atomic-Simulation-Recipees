import pytest
from ase.build import bulk


@pytest.mark.xfail('TODO')
@pytest.mark.acceptance_test
def test_bandstructure_gpaw(asr_tmpdir):
    from asr.c2db.bandstructure import main
    atoms = bulk('Si')

    pathspec = 'GX'
    npoints = 4

    result = main(
        atoms,
        calculator={
            'name': 'gpaw',
            'kpts': (2, 2, 2),
            'mode': 'pw',
        },
        npoints=npoints,
        kptpath=pathspec,
    )

    bs_soc = result['bs_nosoc']
    path = bs_soc['path']
    assert path.path == pathspec
    assert npoints == len(path.kpts)
