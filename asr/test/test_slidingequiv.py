import pytest
from asr.utils.slidingequivalence import mod, equiv_w_vector
from asr.utils.slidingequivalence import equiv_vector, ElementSet
from asr.utils.slidingequivalence import Material, slide_equivalent
from asr.utils.slidingequivalence import get_slide_vector
import numpy as np
from ase import Atoms


@pytest.mark.ci
def test_mod_function(asr_tmpdir):
    np.random.seed(42)
    for _ in range(100):
        L = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        N = np.random.randint(1, 100)

        assert np.allclose(mod(np.array([L, M, N]), 1), 0.0)

    for _ in range(100):
        L = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        N = np.random.randint(1, 100)
        r = (np.random.rand(3) - 0.5) * 5
        r2 = np.array([r[0] + L, r[1] + M, r[2] + N])
        assert np.allclose(mod(r - r2, 1.0), 0.0), f"r: {r}, r2: {r2}"

    # This is from a specific, randomly generated case
    # where the mod function failed
    r1 = np.array([1.26713637, 1.00347767, 0.0])
    r2 = np.array([32.26713637, 9.00347767, 0.0])
    assert np.allclose(mod(r1 - r2, 1.0), 0.0)


def genrandomset():
    np.random.seed(42 + 42)
    n = np.random.randint(1, 10)
    return ElementSet([True for _ in range(n)],
                      [(np.random.rand(3) - 0.5) * 2 for _ in range(n)])


@pytest.mark.ci
def test_equiv_w_vector_special_cases(asr_tmpdir):
    special_cases = [(np.array([1.55503299, 1.67613917, 0.0]),
                      ElementSet([True, True],
                                 [np.array([0, 0, 0.0]), np.array([0.5, 0.0, 0.0])]),
                      ElementSet([True, True],
                                 [np.array([1.55503299, 0.67613917, 0.0]),
                                  np.array([1.05503299, 0.67613917, 0.0])]),
                      True)
                     ]
    for v, s1, s2, expected in special_cases:
        assert equiv_w_vector(v, s1, s2) == expected


@pytest.mark.ci
def test_equiv_w_vector_symmetry(asr_tmpdir):
    np.random.seed(42 * 42)
    for _ in range(100):
        N = np.random.randint(1, 10)
        s1 = ElementSet([True for _ in range(N)],
                        [np.random.rand(3) for _ in range(N)])
        s2 = ElementSet([True for _ in range(N)],
                        [np.random.rand(3) for _ in range(N)])
        v = np.random.rand(3)
        assert equiv_w_vector(v, s1, s2) == equiv_w_vector(v, s2, s1)
        vinteger = np.array([1, 1, 1])
        assert equiv_w_vector(vinteger, s1, s2) == equiv_w_vector(vinteger, s2, s1)


@pytest.mark.ci
def test_equiv_w_vector_integer_translation(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    np.random.seed(42)
    for s in ss:
        I1 = np.random.randint(1, 100)
        I2 = np.random.randint(1, 100)
        I3 = np.random.randint(1, 100)
        assert equiv_w_vector(np.array([I1, I2, I3]), s,
                              s), f"I1: {I1}, I2: {I2}, s.pos: {s.positions}"
        assert equiv_w_vector(np.array([0, 0, 0]), s, s)
        r = np.random.rand(3) + 0.1
        assert not equiv_w_vector(r, s, s), f"r:{r}, s: {s.positions}"


@pytest.mark.ci
def test_equiv_w_vector_equivalent_translations(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    np.random.seed(42)
    for i1, s1 in enumerate(ss):
        r = np.random.rand(3) + 0.1
        r2 = r + 1.0
        assert np.allclose(mod(r - r2, 1.0), 0.0)

        ns = s1.copy()
        ns.set_positions(ns.get_positions(), r)

        ns2 = s1.copy()
        ns2.set_positions(ns2.get_positions(), r2)

        assert equiv_w_vector(np.array([0, 0, 0]), ns, ns2)
        assert equiv_w_vector(r, s1, ns)
        assert equiv_w_vector(r2, s1, ns)


@pytest.mark.ci
def test_not_equiv_if_not_movable(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    np.random.seed(41)
    for i1, s1 in enumerate(ss):
        r = np.random.rand(3) + 0.1
        r2 = r + 1.0
        assert np.allclose(mod(r - r2, 1.0), 0.0)

        ns = s1.copy()
        ns.set_positions(ns.get_positions(), r)
        val = ns._positions[0]
        ns._positions[0] = (False, val[1])

        ns2 = s1.copy()
        ns2.set_positions(ns2.get_positions(), r2)

        for i, (_, p) in enumerate(ns2._positions):
            ns2._positions[i] = (False, p)

        assert equiv_w_vector(np.array([0, 0, 0]), ns, ns2)
        assert not equiv_w_vector(r, ns, s1)
        assert not equiv_w_vector(r2, ns, s1)
        assert not equiv_w_vector(r2, ns2, s1)


@pytest.mark.ci
def test_equiv_vector_diff_lengths(asr_tmpdir):
    np.random.seed(123)
    set1 = ElementSet([True, True],
                      [np.random.rand(3) for _ in range(2)])
    set2 = ElementSet([True, True, True, True],
                      [np.random.rand(3) for _ in range(4)])

    vector = equiv_vector(set1, set2)
    assert vector is None


@pytest.mark.ci
def test_equiv_vector_not_equiv(asr_tmpdir):
    np.random.seed(321)
    N = 10
    ss = [ElementSet([True, True],
                     [np.random.rand(3) for _ in range(2)]) for _ in range(N)]
    vs = [(np.random.rand(3), np.random.rand(3)) for _ in range(N)]
    for (v1, v2), s in zip(vs, ss):
        p1, p2 = s.get_positions()
        s2 = ElementSet([True, True],
                        [p1.copy() + v1, p2.copy() + v2])
        actual = equiv_vector(s, s2)
        if np.allclose(v1, v2):
            assert np.allclose(actual, v1)
        else:
            assert actual is None


@pytest.mark.ci
def test_equiv_vector_given_vector(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    np.random.seed(1020)
    def randompos():
        return np.array([(np.random.rand() - 0.5) * 4,
                         (np.random.rand() - 0.5) * 4,
                         0.0])
    vs = [randompos() for _ in range(50)]

    for v, s in zip(vs, ss):
        s2 = s.copy()
        s2.set_positions(s2.get_positions(), v)
        actual = equiv_vector(s, s2)
        actual2 = equiv_vector(s2, s)
        assert actual is not None, f"{s.get_data()}\n{s2.get_data()}"
        assert np.allclose(mod(actual + actual2, 1),
                           0.0), f"1: {actual}, 2: {actual2}, v:{v}, s: {s.positions}"
        assert np.allclose(mod(actual - v, 1), 0.0)


@pytest.mark.ci
def test_material_setup(asr_tmpdir):
    atoms = Atoms("H2", positions=[[0, 0, 0, ], [0, 0, 1]])
    mat = Material([True, True], atoms)

    assert len(mat.sets) == 1
    assert "H" in mat.sets
    positions = mat.sets["H"].get_positions()
    expected = np.array([[0, 0, 0],
                         [0, 0, 1]])
    assert np.allclose(positions, expected)


@pytest.mark.ci
def test_slide_equivalent_to_self(asr_tmpdir):
    ats = Atoms("H2", positions=[[0, 0, 0, ], [0, 0, 1]], cell=(5, 5, 5))
    ats2 = ats.copy()

    assert ats == ats2
    assert slide_equivalent([False, True], ats, ats2) is not None


@pytest.mark.ci
def test_slide_top_H_remove_sets(asr_tmpdir):
    ats1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=(5, 5, 5))
    ats2 = Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 1]], cell=(5, 5, 5))

    assert ats1 != ats2
    assert len(ats1) == len(ats2)

    mat1 = Material([False, True], ats1)
    mat2 = Material([False for _ in range(len(ats2))],
                    ats2)

    assert len(mat1.sets.keys()) == 1
    assert len(mat2.sets.keys()) == 1

    set1 = mat1.sets["H"]

    s1 = set1.removeAt(0)
    assert s1.get_data()[0][0]
    assert np.allclose(s1.get_data()[0][1], np.array(
        [0, 0, 0.2])), f"actual: {s1.get_data()[0][1]}"

    s2 = set1.removeAt(1)
    assert not s2.get_data()[0][0]
    assert np.allclose(s2.get_data()[0][1], np.array([0, 0, 0]))


@pytest.mark.ci
def test_slide_equiv_singleton_set(asr_tmpdir):
    expected = np.array([0, 0.5, 0.0])
    set1 = ElementSet([True], [np.array([0, 0.0, 0.0])])
    set2 = ElementSet([False], [expected])

    vec = equiv_vector(set1, set2)
    assert np.allclose(vec, expected), vec


@pytest.mark.ci
def test_slide_equiv_double_set(asr_tmpdir):
    expected = np.array([0, 0.5, 0.0])
    zero = np.array([0, 0.0, 0.0])
    set1 = ElementSet([False, True], [zero, zero])
    set2 = ElementSet([False, False], [zero, expected])

    vec = equiv_vector(set1, set2)
    assert vec is not None
    assert np.allclose(vec, expected), vec


@pytest.mark.ci
def test_slide_top_H():
    x = 0.5
    cell = 5
    ats1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=(cell, cell, cell))
    ats2 = Atoms("H2", positions=[[0, 0, 0], [x, 0, 1]], cell=(cell, cell, cell))
    expected = np.array([x / cell, 0, 0])
    assert ats1 != ats2
    assert len(ats1) == len(ats2)

    mat1 = Material([False, True], ats1)
    mat2 = Material([False for _ in range(len(ats2))],
                    ats2)

    vec = equiv_vector(mat1.sets["H"], mat2.sets["H"])

    assert vec is not None
    assert np.allclose(np.abs(vec), expected), vec


@pytest.mark.ci
def test_slide_top_atoms(asr_tmpdir):
    ats = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=(5, 5, 5))
    ats2 = Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 1]], cell=(5, 5, 5))

    assert ats != ats2
    assert len(ats) == len(ats2)

    assert slide_equivalent([False, True], ats, ats2) is not None


@pytest.mark.ci
def test_mos2_slided(asr_tmpdir):
    bottom_pos = np.array([[0, 0, 9.06],
                           [1.59, 0.92, 10.63],
                           [1.59, 0.92, 7.5]])
    cell = np.array([[3.18, 0.0, 0.0],
                     [-1.59, 2.76, 0.],
                     [0, 0., 18.13]])
    cell[2, 2] *= 2
    top_pos = bottom_pos.copy()
    top_pos[:, 2] += 7.5
    bottom = Atoms("MoS2", positions=bottom_pos, cell=cell)
    top = Atoms("MoS2", positions=top_pos, cell=cell)
    bilayer = bottom + top

    slided_top_pos = bottom_pos.copy()
    slided_top_pos[:, 2] += 7.5
    slided_top_pos[:, 0] += 5
    bottom = Atoms("MoS2", positions=bottom_pos, cell=cell)
    slided_top = Atoms("MoS2", positions=slided_top_pos, cell=cell)
    slided_bilayer = bottom + slided_top

    indices = [False, False, False, True, True, True]

    assert slide_equivalent(indices, slided_bilayer, bilayer) is not None
    assert slide_equivalent(indices, bilayer, slided_bilayer) is not None


def invert(atoms):
    pos = atoms.positions

    mean_z = np.mean(pos[:, 2])
    assert np.allclose(mean_z, 9.2203), mean_z

    relative_z = pos.copy()
    relative_z[:, 2] = relative_z[:, 2] - mean_z

    inverted_z = relative_z.copy()
    inverted_z[:, 2] *= -1

    inverted_z[:, 2] += mean_z
    assert (inverted_z[:, 2] >= 0).all()

    atoms2 = atoms.copy()
    atoms2.set_positions(inverted_z)

    dvec = atoms.positions[0] - atoms2.positions[2]
    atoms2.positions[:] += dvec
    atoms2.wrap()

    return atoms2


@pytest.mark.ci
def test_mos2_not_slide_equiv(asr_tmpdir):
    bottom_pos = np.array([[0, 0, 9.06],
                           [1.59, 0.92, 10.63],
                           [1.59, 0.92, 7.5]])
    cell = np.array([[3.18, 0.0, 0.0],
                     [-1.59, 2.76, 0.],
                     [0, 0., 18.13]])
    cell[2, 2] *= 2
    top_pos = bottom_pos.copy()
    top_pos[:, 2] += 7.5
    bottom = Atoms("MoS2", positions=bottom_pos, cell=cell)
    top = Atoms("MoS2", positions=top_pos, cell=cell)
    bilayer = bottom + top

    not_top_pos = np.array([[0, 0, 9.06],
                            [0.59, 0.92, 10.63],
                            [1.59, 2.92, 7.5]])
    not_top_pos[:, 2] += 7.5
    bottom = Atoms("MoS2", positions=bottom_pos, cell=cell)
    not_slided_top = Atoms("MoS2", positions=not_top_pos, cell=cell)
    not_slided_bilayer = bottom + not_slided_top

    indices = [False, False, False, True, True, True]

    assert not_slided_bilayer != bilayer
    assert len(not_slided_bilayer) == len(bilayer)

    assert slide_equivalent(indices, not_slided_bilayer, bilayer) is None
    assert slide_equivalent(indices, bilayer, not_slided_bilayer) is None


@pytest.mark.ci
def test_one_immovable_set(asr_tmpdir):
    ats1 = Atoms("CH", positions=[[0, 0, 0],
                                  [0, 0, 1]], cell=(5, 5, 5))
    ats2 = Atoms("CH", positions=[[0, 0, 0],
                                  [0, 1, 1]], cell=(5, 5, 5))

    v = slide_equivalent([False, True], ats1, ats2)

    assert v is not None
    assert np.allclose(np.array([0, 1 / 5, 0]), v)


@pytest.mark.ci
def test_same_atoms(asr_tmpdir):
    bottom1 = Atoms("H2O", positions=[[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]], cell=(5, 5, 5))
    top1 = bottom1.copy()
    bottom2 = bottom1.copy()
    top2 = bottom1.copy()

    v = get_slide_vector(bottom1, top1, bottom2, top2, np.zeros(3), np.zeros(3))
    assert v is not None
    assert np.allclose(v, np.zeros(3)), v


@pytest.mark.ci
def test_same_misaligned(asr_tmpdir):
    bottom1 = Atoms("H2O", positions=[[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]], cell=(5, 5, 5))
    top1 = bottom1.copy()
    bottom2 = bottom1.copy()

    top2 = Atoms("H2O", positions=[[0, 1, 0],
                                   [0, 1, 1],
                                   [0, 2, 0]], cell=(5, 5, 5))

    v = get_slide_vector(bottom1, top1, bottom2, top2, np.zeros(3), np.zeros(3))
    assert v is not None
    assert np.allclose(v, np.array([0, 1 / bottom1.cell[1, 1], 0])), v


@pytest.mark.ci
def test_not_same_bottom_atoms(asr_tmpdir):
    bottom1 = Atoms("H2O", positions=[[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]], cell=(5, 5, 5))
    top1 = bottom1.copy()
    bottom2 = Atoms("HCO", positions=[[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]], cell=(5, 5, 5))

    top2 = bottom1.copy()

    v = get_slide_vector(bottom1, top1, bottom2, top2, np.zeros(3), np.zeros(3))
    assert v is None, v


@pytest.mark.ci
def test_not_same_top_atoms(asr_tmpdir):
    bottom1 = Atoms("H2O", positions=[[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]], cell=(5, 5, 5))
    top1 = bottom1.copy()
    bottom2 = bottom1.copy()
    top2 = Atoms("HCO", positions=[[0, 0, 0],
                                   [0, 0, 1],
                                   [0, 1, 0]], cell=(5, 5, 5))

    v = get_slide_vector(bottom1, top1, bottom2, top2, np.zeros(2), np.zeros(2))
    assert v is None, v
