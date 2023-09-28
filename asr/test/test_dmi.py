import pytest
import numpy as np
from .materials import Agchain, Fe


@pytest.mark.ci
@pytest.mark.parametrize('test_material', [Agchain, Fe])
@pytest.mark.parametrize('n', [2, [0, 0, 3], 13, [2, 0, 7]])
def test_dmi_integration(asr_tmpdir, mockgpaw, get_webcontent, test_material, n):
    """Test of dmi recipe."""
    from asr.dmi import prepare_dmi, main
    from ase.parallel import world

    test_material.write('structure.json')
    calculator = {"name": "gpaw",
                  "mode": {"mode": "pw", "ecut": 300},
                  "xc": 'LDA',
                  "symmetry": 'off',
                  "experimental": {'soc': False},
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
@pytest.mark.parametrize('n', [2, [0, 1, 3], 13, [2, 0, 7]])
@pytest.mark.parametrize('density', [4.0, 22.0])
def test_findOrthoNN(test_material, n, density):
    from ase.dft.kpoints import monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.dmi import findOrthoNN, kgrid_to_qgrid

    sizes, offsets = kpts2sizeandoffsets(atoms=test_material,
                                         density=density,
                                         gamma=True)

    kpts_kc = monkhorst_pack(sizes) + offsets
    kpts_nqc = findOrthoNN(kpts_kc, test_material.pbc, npoints=n)

    for i, k_qc in enumerate(kpts_nqc):
        q_qc = kgrid_to_qgrid(k_qc)

        dq = q_qc[::2] - q_qc[1::2]
        sign_correction = np.sign(np.sum(dq, axis=-1))
        dq = (dq.T * sign_correction).T
        assert (dq >= 0.).all(), 'Sign correction failed, found negative dq'
