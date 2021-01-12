import numpy as np
import pytest
from .fixtures import various_object_types
from asr.core.selection import Selection


@pytest.mark.ci
def test_selection(various_object_types):
    sel = Selection()
    assert sel.matches(various_object_types)


obj1 = various_object_types
obj2 = various_object_types


@pytest.mark.ci
def test_selection_matches_equality_comparison(obj1, obj2):
    sel = Selection(obj=obj1)
    obj = dict(obj=obj2)
    is_match = sel.matches(obj)

    if isinstance(obj1, np.ndarray):
        obj1 = obj1.tolist()

    if isinstance(obj2, np.ndarray):
        obj2 = obj2.tolist()

    assert is_match == (obj1 == obj2)


@pytest.mark.ci
def test_int_does_not_match_str():

    sel = Selection(id=0)
    obj = dict(id='abc')

    assert not sel.matches(obj)
