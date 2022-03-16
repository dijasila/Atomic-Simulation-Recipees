import pytest


@pytest.mark.skip
@pytest.mark.acceptance_test
def test_relax_fe_gpaw(asr_tmpdir):
    from asr.relax import main
    from ase.io import read
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
    Fe.write('unrelaxed.json')
    main()
    relaxed = read('structure.json')
    magmoms = relaxed.get_initial_magnetic_moments()
    assert magmoms[0] == pytest.approx(magmom, abs=0.1)
