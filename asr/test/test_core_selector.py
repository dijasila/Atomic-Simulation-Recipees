import numpy as np
import pytest
from .fixtures import various_object_types
from asr.core.selector import Selector


@pytest.mark.ci
def test_selector(various_object_types):
    sel = Selector()
    assert sel.matches(various_object_types)


obj1 = various_object_types
obj2 = various_object_types


@pytest.mark.ci
def test_selector_matches_equality_comparison(obj1, obj2):
    sel = Selector(obj=Selector.EQUAL(obj1))
    obj = dict(obj=obj2)
    is_match = sel.matches(obj)

    if isinstance(obj1, np.ndarray):
        obj1 = obj1.tolist()

    if isinstance(obj2, np.ndarray):
        obj2 = obj2.tolist()

    assert is_match == (obj1 == obj2)


@pytest.mark.ci
def test_selector_specification_via_attributes():
    sel = Selector()
    sel.a.b = Selector.EQUAL(2)

    assert sel.matches(dict(a=dict(b=2)))


# @pytest.mark.ci
# def test_match_nested_attribues():
#     sel = Selector({'a.b': equal(2)})
#     sel.a.b = equal(2)


@pytest.mark.ci
def test_int_does_not_match_str():
    sel = Selector(id=Selector.EQUAL(0))
    obj = dict(id="abc")

    assert not sel.matches(obj)
