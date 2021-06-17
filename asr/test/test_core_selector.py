import numpy as np
import pytest
from .fixtures import various_object_types
from .materials import BN, Ag
from asr.core.selector import Selector


@pytest.mark.ci
def test_selector(various_object_types):
    sel = Selector()
    assert sel.matches(various_object_types)


obj1 = various_object_types
obj2 = various_object_types


@pytest.mark.ci
def test_selector_matches_equality_comparison(obj1, obj2):
    sel = Selector()
    sel.obj = sel.EQUAL(obj1)
    obj = dict(obj=obj2)
    is_match = sel.matches(obj)

    if isinstance(obj1, np.ndarray):
        obj1 = obj1.tolist()

    if isinstance(obj2, np.ndarray):
        obj2 = obj2.tolist()

    assert is_match == (obj1 == obj2)


@pytest.mark.ci
def test_selector_with_attribute_access():
    class TestObject:
        pass

    sel = Selector()
    sel.obj = sel.EQUAL(obj1)

    test_object = TestObject()
    test_object.obj = obj2
    assert sel.matches(test_object)


@pytest.mark.ci
def test_selector_specification_via_nested_attributes():
    sel = Selector()
    sel.a.b.c = sel.EQUAL(2)

    assert sel.matches(dict(a=dict(b=dict(c=2))))


@pytest.mark.ci
def test_int_does_not_match_str():
    sel = Selector()
    sel.id = sel.EQUAL(0)
    obj = dict(id="abc")

    assert not sel.matches(obj)


@pytest.mark.ci
def test_selector_missing_attribute():
    sel = Selector()
    sel.attr = sel.EQUAL(0)
    obj = dict()

    assert not sel.matches(obj)


@pytest.mark.ci
def test_selector_specification_constructor():
    kwargs = dict(id=Selector.EQ(0))
    sel = Selector(**kwargs)
    obj = dict(id=0)

    assert sel.matches(obj)


@pytest.mark.ci
def test_selector_str():
    sel = Selector()
    sel.attr = sel.EQUAL(0)

    assert str(sel) == 'Selector(attr=equal(0))'


@pytest.mark.ci
def test_selector_repr():
    sel = Selector()
    sel.attr = sel.EQUAL(0)

    assert repr(sel) == str(sel)


@pytest.mark.ci
@pytest.mark.parametrize(
    'cmp_meth,obj1,obj2,result',
    [
        (Selector.EQ, 2, 2, True),
        (Selector.EQ, 2.01, 2.01, True),
        (Selector.EQ, 1, 2, False),
        (Selector.EQ, 2, 1, False),
        (Selector.IS, None, None, True),
        (Selector.IS, 2, 2, True),
        (Selector.IS, 2, 1, False),
        (Selector.LT, 1, 257, True),
        (Selector.LT, 257, 1, False),
        (Selector.LT, 257, 257, False),
        (Selector.LTE, 257, 257, True),
        (Selector.LTE, 257, 1, False),
        (Selector.LTE, 1, 257, True),
        (Selector.GT, 257, 257, False),
        (Selector.GT, 257, 1, True),
        (Selector.GT, 1, 257, False),
        (Selector.GTE, 257, 257, True),
        (Selector.GTE, 257, 1, True),
        (Selector.GTE, 1, 257, False),
        (Selector.APPROX, 1, 1.0001, True),
        (Selector.APPROX, 1, 1.01, False),
        (Selector.APPROX, 1.0001, 1, True),
        (Selector.APPROX, 1.01, 1, False),
        (Selector.ATOMS_EQUAL_TO, BN, BN, True),
        (Selector.ATOMS_EQUAL_TO, BN, Ag, False),
        (Selector.ATOMS_EQUAL_TO, Ag, BN, False),
        (Selector.ATOMS_EQUAL_TO, Ag, Ag, True),
    ],
)
def test_selector_comparison_methods(cmp_meth, obj1, obj2, result):
    assert cmp_meth(obj1)(obj2) == result
