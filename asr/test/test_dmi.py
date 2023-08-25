import pytest
import numpy as np
from typing import List
from .materials import Agchain, Fe


@pytest.mark.ci
@pytest.mark.parametrize('test_material', [Agchain, Fe])
@pytest.mark.parametrize('n', [2, [0, 0, 3], 13, [2, 0, 7]])
def test_dmi(asr_tmpdir_w_params, mockgpaw, get_webcontent, test_material, n):
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

    prep = prepare_dmi(calculator, n=n)
    result = main()

    if world.size == 1:
        content = get_webcontent()
        print(content)
        assert False

@pytest.mark.ci
@pytest.mark.parametrize('n', [2, [0, 1, 3], 13, [2, 0, 7]])
@pytest.mark.parametrize('density', [4.0, 22.0])
def test_findOrthoNN(test_material, n, density):
    from ase.dft.kpoints import monkhorst_pack
    from ase.calculators.calculator import kpts2sizeandoffsets
    from asr.dmi import findOrthoNN

    sizes, offsets = kpts2sizeandoffsets(atoms=test_material,
                                         density=density,
                                         gamma=True)

    kpts_kc = monkhorst_pack(sizes) + offsets
    qpts_nqc = findOrthoNN(kpts_kc, test_material.pbc, npoints=n)

    q_nqc = []
    for i, q_qc in enumerate(qpts_nqc):
        q_qc *= 2
        q_qc = q_qc[np.linalg.norm(q_qc, axis=-1) <= 0.5]
        even_q = int(np.floor(len(q_qc) / 2) * 2)
        q_qc = q_qc[:even_q]

        dq = q_qc[::2] - q_qc[1::2]
        sign_correction = np.sign(np.sum(dq, axis=-1))
        dq = (dq.T * sign_correction).T
        assert (dq >= 0.).all(), 'Sign correction failed, found negative dq'
        
def old_findOrthoNN(kpts: List[float], pbc: List[bool], npoints: int = 2, eps: float = 0):
    '''
    Given a list of kpoints, we find the points along vectors [1,0,0], [0,1,0], [0,0,1]
    and search through them ordered on the distance to the origin. Vectors along the
    postive axis will appear first.
    '''
    # Warning, might not find inversion symmetric points if k-points are not symmetric
    from scipy.spatial import cKDTree

    # Calculate distance-ordered indices from the (eps postive) origin
    _, indices = cKDTree(kpts).query([eps, eps, eps], k=len(kpts))
    indices = indices[1:]

    N = sum(pbc)
    periodic_directions = np.where(pbc)[0]

    orthNN = [[], [], []]#[:N]
    for direction in periodic_directions: # np.arange(N):
        orthoDirs = [(direction + 1) % 3, (direction + 2) % 3]
        i = 0
        for j, idx in enumerate(indices):
            # Check if point lies on a line x, y or z
            if np.isclose(kpts[idx][orthoDirs[0]], 0) and \
               np.isclose(kpts[idx][orthoDirs[1]], 0):
                orthNN[direction].append(kpts[idx])
                i += 1
                if i == npoints:
                    break

    orthNN = [NN for j, NN in enumerate(orthNN) if j in np.where(pbc)[0]]
    shape = [np.shape(orthNN[j]) for j in range(N)]
    assert (0,) not in shape, \
        f'No k-points in some periodic direction(s), out.shape = {shape}'
    assert shape != [], 'No k-points were found'
    assert all([(npoints, 3) == np.shape(orthNN[j]) for j in range(N)]), \
        f'Missing k-points in some periodic direction(s), out.shape = {shape}'
    # This test is incompatible with len(pbc) = 2, while it works fine otherwise
    # assert not all([all(np.dot(orthNN[i], pbc) == 0) for i in range(N)]), \
    #     f'The k-points found are in a non-periodic direction'
    return np.round(np.array(orthNN), 16)
