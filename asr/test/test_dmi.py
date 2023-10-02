import pytest
import numpy as np
from .materials import Agchain, Fe


@pytest.mark.ci
@pytest.mark.parametrize('test_material', [Agchain, Fe])
@pytest.mark.parametrize('n', [2, [0, 0, 3], 13, [2, 0, 7]])
@pytest.mark.skipif(True, reason='TODO: mockgpaw of new GPAW')
def test_dmi_mock(asr_tmpdir, get_webcontent, test_material, n):  # mockgpaw,
    """Test of dmi recipe."""
    from asr.dmi import prepare_dmi, main
    from ase.parallel import world

    test_material.write('structure.json')
    calculator = {"mode": {"name": "pw", "ecut": 300},
                  "xc": 'LDA',
                  "symmetry": 'off',
                  "soc": False,
                  "parallel": {"domain": 1, 'band': 1},
                  "kpts": {"density": 6, "gamma": True}}

    prepare_dmi(calculator, n=n)
    main()

    if world.size == 1:
        content = get_webcontent()
        if type(n) is list:
            number_of_D = sum(pbc for i, pbc in enumerate(test_material.pbc)
                              if n[i] != 0)
        else:
            number_of_D = sum(test_material.pbc)

        for i in range(number_of_D):
            assert f"D(q<sub>a{i + 1}</sub>)(meV/Å<sup>-1</sup>)" in content, content


@pytest.mark.ci
def test_dmi_integration(asr_tmpdir, get_webcontent):
    """Test of dmi recipe."""
    from asr.dmi import prepare_dmi, main
    from ase.parallel import world
    from ase import Atoms
    from numpy import array

    atoms = Atoms('H', cell=[3, 3, 3], pbc=[False, True, False])
    atoms.center()
    atoms.write('structure.json')
    magmoms = array([[1, 0, 0]])

    calculator = {"mode": {"name": "pw", "ecut": 300},
                  "xc": 'LDA',
                  "symmetry": 'off',
                  "soc": False,
                  "magmoms": magmoms,
                  "parallel": {"domain": 1, 'band': 1},
                  "kpts": {"density": 6, "gamma": True}}

    prepare_dmi(calculator, [0, 2, 0])
    main()

    if world.size == 1:
        content = get_webcontent()
        assert f"D(q<sub>a{1}</sub>)(meV/Å<sup>-1</sup>)" in content, content


@pytest.mark.ci
@pytest.mark.parametrize('n', [2, [0, 1, 3], 13, [2, 0, 7]])
@pytest.mark.parametrize('density', [4.0, 22.0])
def test_find_ortho_nn(test_material, n, density):
    from ase.dft.kpoints import monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.dmi import find_ortho_nn, kgrid_to_qgrid

    sizes, offsets = kpts2sizeandoffsets(atoms=test_material,
                                         density=density,
                                         gamma=True)

    kpts_kc = monkhorst_pack(sizes) + offsets
    kpts_nqc = find_ortho_nn(kpts_kc, test_material.pbc, npoints=n)

    for i, k_qc in enumerate(kpts_nqc):
        q_qc = kgrid_to_qgrid(k_qc)

        dq = q_qc[::2] - q_qc[1::2]
        sign_correction = np.sign(np.sum(dq, axis=-1))
        dq = (dq.T * sign_correction).T
        assert (dq >= 0.).all(), 'Sign correction failed, found negative dq'
