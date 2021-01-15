import pytest
import numpy as np

from asr.core.serialize import JSONSerializer

serializer = JSONSerializer()


@pytest.mark.ci
def test_encode_decode(various_object_types):

    serialized = serializer.serialize(various_object_types)
    deserialized = serializer.deserialize(serialized)

    # np.arrays treat == differently than most objects
    # in particular they don't return bool
    if isinstance(various_object_types, np.ndarray):
        various_object_types = various_object_types.tolist()
        deserialized = deserialized.tolist()

    assert various_object_types == deserialized
