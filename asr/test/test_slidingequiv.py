import pytest
from asr.utils.slidingequivalence import mod, equiv_w_vector, equiv_vector, ElementSet, Material
import numpy as np
from ase import Atoms
    

@pytest.mark.ci
def test_mod_function(asr_tmpdir):
    for _ in range(100):
        M = np.random.randint(1, 100)
        N = np.random.randint(1, 100)
        assert np.allclose(mod(np.array([M, N]), 1), 0.0)

    for _ in range(100):
        M = np.random.randint(1, 100)
        N = np.random.randint(1, 100)
        r = (np.random.rand(2) - 0.5) * 5
        r2 = np.array([r[0] + M, r[1] + N])
        assert np.allclose(mod(r - r2, 1.0), 0.0), f"r: {r}, r2: {r2}"


    r1 = np.array([1.26713637, 1.00347767])
    r2 = np.array([32.26713637, 9.00347767])
    assert np.allclose(mod(r1-r2, 1.0), 0.0)



def genrandomset():
    n = np.random.randint(1, 10)
    return ElementSet([(np.random.rand(2) - 0.5) * 2 for _ in range(n)])


@pytest.mark.ci
def test_equiv_w_vector_special_cases(asr_tmpdir):
    special_cases = [(np.array([1.55503299, 1.67613917]),
                      ElementSet([np.array([0,0]), np.array([0.5, 0.0])]),
                      ElementSet([np.array([1.55503299, 0.67613917]), np.array([1.05503299, 0.67613917])]),
                      True)
                 ]
    for v, s1, s2, expected in special_cases:
        assert equiv_w_vector(v, s1, s2) == expected



@pytest.mark.ci
def test_equiv_w_vector_symmetry(asr_tmpdir):
    for _ in range(100):
        N = np.random.randint(1, 10)
        s1 = ElementSet([np.random.rand(2) for _ in range(N)])
        s2 = ElementSet([np.random.rand(2) for _ in range(N)])
        v = np.random.rand(2)
        assert equiv_w_vector(v, s1, s2) == equiv_w_vector(v, s2, s1)
        vinteger = np.array([1, 1])
        assert equiv_w_vector(vinteger, s1, s2) == equiv_w_vector(vinteger, s2, s1)

@pytest.mark.ci
def test_equiv_w_vector_integer_translation(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]

    for s in ss:
        I1 = np.random.randint(1, 100)
        I2 = np.random.randint(1, 100)
        assert equiv_w_vector(np.array([I1, I2]), s, s), f"I1: {I1}, I2: {I2}, s.pos: {s.positions}"
        assert equiv_w_vector(np.array([0, 0]), s, s)
        r = np.random.rand(2) + 0.1
        assert not equiv_w_vector(r, s, s), f"r:{r}, s: {s.positions}"


@pytest.mark.ci
def test_equiv_w_vector_equivalent_translations(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    
    for i1, s1 in enumerate(ss):
        r = np.random.rand(2) + 0.1
        r2 = r + 1.0
        assert np.allclose(mod(r - r2, 1.0), 0.0)
        ns = ElementSet(s1.positions.copy())
        ns.positions = [p.copy() + r for p in ns.positions]
        ns2 = ElementSet(s1.positions.copy())
        ns2.positions = [p.copy() + r2 for p in ns2.positions]

        assert equiv_w_vector(np.array([0, 0]), ns, ns2)
        assert equiv_w_vector(r, s1, ns)
        assert equiv_w_vector(r2, s1, ns)

@pytest.mark.ci
def test_equiv_vector_diff_lengths(asr_tmpdir):
    set1 = ElementSet([np.random.rand(2) for _ in range(2)])
    set2 = ElementSet([np.random.rand(2) for _ in range(4)])

    vector = equiv_vector(set1, set2)
    assert vector is None

@pytest.mark.ci
def test_equiv_vector_not_equiv(asr_tmpdir):
    N = 10
    ss = [ElementSet([np.random.rand(2) for _ in range(2)]) for _ in range(N)]
    vs = [(np.random.rand(2), np.random.rand(2)) for _ in range(N)]
    for (v1, v2), s in zip(vs, ss):
        p1, p2 = s.positions
        s2 = ElementSet([p1.copy() + v1, p2.copy() + v2])
        actual = equiv_vector(s, s2)
        if np.allclose(v1, v2):
            assert np.allclose(actual, v1)
        else:
            assert actual is None

@pytest.mark.ci
def test_equiv_vector_given_vector(asr_tmpdir):
    ss = [genrandomset() for _ in range(50)]
    vs = [(np.random.rand(2) - 0.5) * 4 for _ in range(50)]

    for v, s in zip(vs, ss):
        s2 = ElementSet([p.copy() + v for p in s.positions])
        actual = equiv_vector(s, s2)
        actual2 = equiv_vector(s2, s)
        assert np.allclose(mod(actual + actual2, 1), 0.0), f"1: {actual}, 2: {actual2}, v:{v}, s: {s.positions}"
        assert np.allclose(mod(actual - v, 1), 0.0)


@pytest.mark.ci
def test_slide_equivalent_to_self(asr_tmpdir):
    ats = Atoms("H2", positions=[[0, 0, 0,], [0, 0, 1]])
    ats2 = ats.copy()

    assert slide_equivalent(ats, ats2)
    
