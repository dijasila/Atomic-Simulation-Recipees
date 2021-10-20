import copy
import ase
import simplejson as json
import typing
import pathlib
from .results import obj_to_id
import datetime
import numpy as np


class ASRJSONEncoder(json.JSONEncoder):

    def default(self, obj) -> dict:

        if hasattr(obj, '__dict__'):
            cls_id = obj_to_id(obj.__class__)
            obj = {'cls_id': cls_id, '__dict__':
                   copy.copy(obj.__dict__)}
            return obj

        if isinstance(obj, set):
            return {
                '__asr_type__': 'set',
                'value': list(obj),
            }
        elif isinstance(obj, np.ndarray):
            # Out tuple serialization causes trouble with
            # ASE's numpy serialization so here we implement our
            # own to avoid conflict with ASE.
            flatobj = obj.ravel()
            if np.iscomplexobj(obj):
                flatobj.dtype = obj.real.dtype
            return {'__ndarray__': [list(obj.shape),
                                    obj.dtype.name,
                                    flatobj.tolist()]}
        elif isinstance(obj, datetime.datetime):
            return {
                '__asr_type__': 'datetime.datetime',
                'value': obj.isoformat(),
            }
        elif isinstance(obj, tuple):
            return {
                '__asr_type__': 'tuple',
                'value': list(obj),
            }
        elif isinstance(obj, pathlib.Path):
            return {
                '__asr_type__': 'pathlib.Path',
                'value': str(obj),
            }
        return ase.io.jsonio.MyEncoder.default(self, obj)


def json_hook(json_object: dict):
    from asr.core.results import get_object_matching_obj_id
    from ase.io.jsonio import object_hook

    if 'cls_id' in json_object:
        assert '__dict__' in json_object
        cls = get_object_matching_obj_id(json_object['cls_id'])
        obj = cls.__new__(cls)
        obj.__dict__.update(json_object['__dict__'])
        return obj

    asr_type = json_object.get('__asr_type__')

    if asr_type is not None:
        value = json_object['value']
        if asr_type == 'set':
            return set(value)
        if asr_type == 'datetime.datetime':
            return datetime.datetime.fromisoformat(value)
        elif asr_type == 'tuple':
            return tuple(value)
        elif asr_type == 'pathlib.Path':
            return pathlib.Path(value)

    return object_hook(json_object)


class JSONSerializer:

    encoder = ASRJSONEncoder(tuple_as_array=False).encode
    decoder = json.JSONDecoder(object_hook=json_hook).decode

    def serialize(self, obj) -> str:
        """Serialize object to JSON."""
        return self.encoder(obj)

    def deserialize(self, serialized: str) -> typing.Any:
        """Deserialize json object."""
        return self.decoder(serialized)
