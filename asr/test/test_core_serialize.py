import pytest
import numpy as np

from asr.core.serialize import JSONSerializer

serializer = JSONSerializer()


def serialize_deserialize(obj):
    serialized = serializer.serialize(obj)
    deserialized = serializer.deserialize(serialized)

    return deserialized


@pytest.mark.ci
def test_encode_decode(various_object_types):

    deserialized = serialize_deserialize(various_object_types)

    # np.arrays treat == differently than most objects
    # in particular they don't return bool
    if isinstance(various_object_types, np.ndarray):
        various_object_types = various_object_types.tolist()
        deserialized = deserialized.tolist()

    assert various_object_types == deserialized


@pytest.mark.ci
def test_encode_decode_external_file(external_file):
    deserialized = serialize_deserialize(external_file)

    assert external_file == deserialized
