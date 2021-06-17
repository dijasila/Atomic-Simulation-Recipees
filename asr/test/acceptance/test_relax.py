import pytest


@pytest.mark.skip
@pytest.mark.acceptance_test
def test_relax_fe_gpaw(asr_tmpdir):
    from asr.relax import main
    from ase import Atoms
    a = 1.41973054
    magmom = 2.26739285
    Fe = Atoms('Fe',
               positions=[[0., 0., 0.]],
               cell=[[-a, a, a],
                     [a, -a, a],
                     [a, a, -a]],
               magmoms=[magmom],
               pbc=True)

    parameters = dict(
        name='gpaw',
        mode={'name': 'pw', 'ecut': 200.0},
        kpts=[2, 2, 2],
    )

    record = main(Fe,
                  calculator=parameters,
                  fmax=0.05)

    relaxed = record.result['atoms']
    magmoms = relaxed.get_initial_magnetic_moments()
    assert magmoms[0] == pytest.approx(magmom, abs=0.1)
