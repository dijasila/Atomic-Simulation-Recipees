from asr.bandstructure import main
from ase.build import bulk


def test_bandstructure_gpaw(asr_tmpdir):
    atoms = bulk('Si')

    pathspec = 'GX'
    npoints = 4

    record = main(
        atoms,
        calculator={
            'name': 'gpaw',
            'kpts': (2, 2, 2),
            'mode': 'pw',
        },
        npoints=npoints,
        kptpath=pathspec,
    )

    result = record.result
    bs_soc = result['bs_nosoc']
    path = bs_soc['path']
    assert path.path == pathspec
    assert npoints == len(path.kpts)
