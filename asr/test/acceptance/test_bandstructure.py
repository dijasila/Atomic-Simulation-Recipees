import pytest
from ase.build import bulk


@pytest.mark.acceptance_test
def test_bandstructure_gpaw(asr_tmpdir):
    from asr.c2db.bandstructure import BS
    from asr.c2db.gs import GS

    atoms = bulk('Si')

    pathspec = 'GX'
    npoints = 4

    calculator = {
        'name': 'gpaw',
        'kpts': (2, 2, 2),
        'mode': 'pw'}

    gs = GS(atoms=atoms, calculator=calculator)

    bs = BS(gs=gs, kptpath=pathspec,
            npoints=npoints)
    result = bs.post

    bs_soc = result['bs_nosoc']
    path = bs_soc['path']
    assert path.path == pathspec
    assert npoints == len(path.kpts)
